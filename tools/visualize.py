import argparse
import copy
# 本脚本只导出图片；在 mmcv/mmdet3d 间接导入 pyplot 前固定无窗口后端。
# 避免 Matplotlib 启动 Qt，与 OpenCV 自带的 xcb 插件发生冲突。
import matplotlib
matplotlib.use("Agg", force=True)

import cv2
import os
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner.fp16_utils import wrap_fp16_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj



# Occ3D 的 0～17 类颜色；17 为 free，255 为忽略区域，不绘制空体素。
OCC_COLORS = np.array([
    [0, 0, 0], [112, 128, 144], [220, 20, 60], [255, 69, 0],
    [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230],
    [47, 79, 79], [255, 140, 0], [255, 99, 71], [0, 207, 191],
    [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135],
    [0, 175, 0], [255, 255, 255],
], dtype=np.uint8)



def save_occupancy_array(path, occupancy):
    # 同时兼容模型输出的 NumPy 数组和 Tensor，保持原始形状、类别值与数据类型。
    if torch.is_tensor(occupancy):
        occupancy = occupancy.detach().cpu().numpy()
    mmcv.mkdir_or_exist(os.path.dirname(path))
    np.save(path, np.asarray(occupancy))


# 保留三维渲染函数供后续使用，当前主流程只导出 npy，不调用该函数。

def render_occupancy(path, occupancy, title):
    # 使用无窗口绘图后端，将占用体素的外露表面绘制为三维彩色网格。
    # 相邻占用体素的内部面不绘制，减少全量体素绘制的时间和内存开销。
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if torch.is_tensor(occupancy):
        occupancy = occupancy.detach().cpu().numpy()
    occupancy = np.asarray(occupancy)
    if occupancy.shape != (200, 200, 16):
        raise ValueError(f"Expected Occ3D shape (200, 200, 16), got {occupancy.shape}")
    occupied = (occupancy >= 0) & (occupancy < 17)
    faces, colors = [], []
    # Occ3D 坐标范围为 x/y [-40,40]、z [-1,5.4]，体素边长 0.4 米。
    origin = np.array([-40., -40., -1.])
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for side in (-1, 1):
            neighbor = np.roll(occupied, -side, axis=axis)
            boundary = [slice(None)] * 3
            boundary[axis] = -1 if side == 1 else 0
            neighbor[tuple(boundary)] = False
            positions = np.argwhere(occupied & ~neighbor)
            corners = np.zeros((4, 3))
            corners[:, axis] = int(side == 1)
            corners[:, other[0]] = [0, 1, 1, 0]
            corners[:, other[1]] = [0, 0, 1, 1]
            faces.append((positions[:, None, :] + corners) * 0.4 + origin)
            colors.append(OCC_COLORS[occupancy[tuple(positions.T)].astype(int)] / 255.)
    fig = Figure(figsize=(10, 9), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    if occupied.any():
        ax.add_collection3d(Poly3DCollection(
            np.concatenate(faces), facecolors=np.concatenate(colors),
            edgecolors="none", linewidths=0,
        ))
    ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-1, 5.4),
           xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
    ax.set_box_aspect((80, 80, 6.4))
    ax.view_init(elev=35, azim=-60)
    fig.tight_layout()
    mmcv.mkdir_or_exist(os.path.dirname(path))
    fig.savefig(path)
    fig.clear()


def save_surround_image(path, metas, bboxes, labels, classes, thickness):
    # 按相机名称定位，避免依赖 info 中六路相机的字典顺序。
    order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    views = {}
    for k, image_path in enumerate(metas["filename"]):
        channel = next((c for c in order if c in Path(image_path).parts), None)
        if channel is None or channel in views:
            raise ValueError(f"Cannot uniquely identify nuScenes camera: {image_path}")
        image = visualize_camera(
            None, mmcv.imread(image_path), bboxes=bboxes, labels=labels,
            transform=metas["lidar2image"][k], classes=classes, thickness=thickness,
        )
        # 统一缩放到 800×450，添加相机名称，六路只保存一张 2×3 图。
        image = cv2.resize(image, (800, 450), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (800, 32), (25, 25, 25), -1)
        cv2.putText(image, channel, (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        views[channel] = image
    if set(views) != set(order):
        raise ValueError("Surround visualization requires all six nuScenes cameras")
    mosaic = np.concatenate([
        np.concatenate([views[c] for c in order[:3]], axis=1),
        np.concatenate([views[c] for c in order[3:]], axis=1),
    ], axis=0)
    mmcv.mkdir_or_exist(os.path.dirname(path))
    mmcv.imwrite(mosaic, path)


def main() -> None:
    # dist.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    # 已保存的逐帧预测根目录，布局为 <pred-dir>/<scene-name>/<sample-token>.pkl。
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.45)
    # 相机检测框默认从 4 像素减为 2 像素，点云框线宽单独设置。
    parser.add_argument("--bbox-thickness", type=int, default=2)
    parser.add_argument("--lidar-bbox-thickness", type=float, default=5.0)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--interval", type=int, default=1)
    args, opts = parser.parse_known_args()
    if args.bbox_thickness < 1 or args.lidar_bbox_thickness <= 0:
        parser.error("bbox thickness must be positive")
    if args.interval < 1:
        parser.error("--interval must be positive")
    if args.pred_dir and args.mode != "pred":
        parser.error("--pred-dir requires --mode pred")
    if args.mode == "pred" and not args.pred_dir and not args.checkpoint:
        parser.error("prediction mode requires --pred-dir or --checkpoint")
    # 只索引目录，不提前加载全部预测，避免重复推理和额外内存占用。
    prediction_files = {}
    if args.pred_dir:
        for path in sorted(Path(args.pred_dir).expanduser().glob("*/*.pkl")):
            if path.stem in prediction_files:
                raise ValueError(f"Duplicate prediction token: {path.stem}")
            prediction_files[path.stem] = path
        if not prediction_files:
            raise FileNotFoundError(f"No scene/sample_token.pkl files in {args.pred_dir}")

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    # torch.cuda.set_device(dist.local_rank())
    # 离线预测读取和真值可视化不执行模型推理，无需选择 CUDA 设备。
    if args.mode == "pred" and not args.pred_dir:
        torch.cuda.set_device(0)

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        # 此脚本使用单进程可视化，避免分布式采样遗漏帧。
        dist=False,
        shuffle=False,
    )

    model = None
    # build the model and load checkpoint
    if args.mode == "pred" and not args.pred_dir:
        model = build_model(cfg.model)
        # fix bug
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        # )
        model = MMDataParallel(model.cuda(), device_ids=[0])
        model.eval()

    frame_id = -1
    for data in tqdm(dataflow):
        frame_id += 1
        if frame_id % args.interval != 0:
            continue
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        frame_out_dir = args.out_dir
        if args.mode == "pred":
            if args.pred_dir:
                # 通过 token 精确匹配帧；允许可视化尚未完成推理的部分结果目录。
                prediction_path = prediction_files.get(metas["token"])
                if prediction_path is None:
                    continue
                outputs = [mmcv.load(str(prediction_path))]
                frame_out_dir = os.path.join(args.out_dir, prediction_path.parent.name)
            else:
                with torch.inference_mode():
                    outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        # 恢复导出 occupancy 数组，供 vis_occ3d.py 进行交互式三维可视化。
        # 预测保存为 <name>.npy，真值保存为 <name>_gt.npy；不再调用三维 PNG 渲染。
        if args.mode == "pred" and "occ_pred" in outputs[0]:
            save_occupancy_array(
                os.path.join(frame_out_dir, "occ", f"{name}.npy"),
                outputs[0]["occ_pred"],
            )
        if "voxel_semantics" in data:
            save_occupancy_array(
                os.path.join(frame_out_dir, "occ", f"{name}_gt.npy"),
                data["voxel_semantics"][0],
            )

        # 原逐相机保存改为内存绘制后拼接，不再创建 camera-0～camera-5 目录。
        if "img" in data:
            save_surround_image(
                os.path.join(frame_out_dir, "surround", f"{name}.png"),
                metas, bboxes, labels, cfg.object_classes, args.bbox_thickness,
            )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(frame_out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                thickness=args.lidar_bbox_thickness,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(frame_out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
