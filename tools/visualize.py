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



#* (11.12) 与 tools/vis_occ3d.py 的语义类别 0～16 配色一致。
# 不直接导入该脚本，避免其 Qt 环境配置影响无窗口批量导图。
OCC_COLORS = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0],
    [0, 150, 245], [0, 255, 255], [200, 180, 0], [255, 0, 0],
    [255, 240, 150], [135, 60, 0], [160, 32, 240], [255, 0, 255],
    [139, 137, 137], [75, 0, 75], [150, 240, 80], [230, 230, 250],
    [0, 175, 0],
], dtype=np.uint8)



def save_occupancy_array(path, occupancy):
    # 同时兼容模型输出的 NumPy 数组和 Tensor，保持原始形状、类别值与数据类型。
    if torch.is_tensor(occupancy):
        occupancy = occupancy.detach().cpu().numpy()
    mmcv.mkdir_or_exist(os.path.dirname(path))
    np.save(path, np.asarray(occupancy))


#* (11.12) 正上方俯视投影：每个 XY 格选最高可见体素；预测和 GT 同图比较。
def occupancy_top_view(occupancy):
    if torch.is_tensor(occupancy):
        occupancy = occupancy.detach().cpu().numpy()
    occupancy = np.asarray(occupancy)
    if occupancy.shape != (200, 200, 16):
        raise ValueError(f"Expected Occ3D shape (200, 200, 16), got {occupancy.shape}")
    visible = (occupancy > 0) & (occupancy < 17)
    has_voxel = visible.any(axis=2)
    top_z = occupancy.shape[2] - 1 - visible[:, :, ::-1].argmax(axis=2)
    labels = np.take_along_axis(occupancy, top_z[..., None], axis=2)[..., 0]
    rgb = np.full((200, 200, 3), 255, dtype=np.uint8)
    rgb[has_voxel] = OCC_COLORS[labels[has_voxel].astype(np.int64)]
    # 数组轴为 X,Y,Z；imshow 的列为 X、行为 Y。
    return rgb.transpose(1, 0, 2)


def render_occupancy_comparison(path, prediction, ground_truth, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    panels = [('Prediction', prediction), ('Ground truth', ground_truth)]
    fig = Figure(figsize=(13, 6.5), dpi=120)
    FigureCanvasAgg(fig)
    for i, (label, occupancy) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 2, i)
        if occupancy is not None:
            ax.imshow(occupancy_top_view(occupancy), origin='lower',
                      extent=(-40, 40, -40, 40), interpolation='nearest')
        else:
            ax.text(0, 0, 'Not available', ha='center', va='center')
        ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel='Ego X (m)',
               ylabel='Ego Y (m)', title=label)
        ax.set_aspect('equal')
    fig.suptitle(title + '\nTop view: highest visible voxel; classes 0, 17 and 255 hidden')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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

    #* (11.12) 可视化加载器固定 BS=1，samples_per_gpu 不属于数据集构造参数。
    # 离线绘图按 token 读取已保存预测，不需要序列采样和增强一致性配置。
    dataset_cfg = copy.deepcopy(cfg.data[args.split])

    def prepare_visual_dataset(config):
        if isinstance(config, list):
            for child in config:
                prepare_visual_dataset(child)
            return
        config.pop('samples_per_gpu', None)
        if config.get('type') in ('NuScenesDatasetOccupancy', 'NuScenesSequenceDataset'):
            config.pop('sequences_split_num', None)
            config.pop('keep_consistent_seq_aug', None)
            if config['type'] == 'NuScenesSequenceDataset':
                config['type'] = 'NuScenesDatasetOccupancy'
        if 'dataset' in config:
            prepare_visual_dataset(config['dataset'])
        if 'datasets' in config:
            prepare_visual_dataset(config['datasets'])

    prepare_visual_dataset(dataset_cfg)
    #* (11.12) 流式权重必须先按序列推理保存，避免本入口退回单帧模型而忽略融合权重。
    if (cfg.get('streaming', {}).get('enabled', False)
            and args.mode == 'pred' and not args.pred_dir):
        parser.error('流式模型请先用 tools/test.py --save-predictions，再用 --pred-dir 可视化')
    dataset = build_dataset(dataset_cfg)
    #* (11.12) 在加载图像、点云及标注之前筛选帧，避免先遍历完整验证集再跳过。
    # interval 仍按原数据集索引计算；预测通过 token 匹配，无需连续读取历史帧。
    selected_indices = list(range(0, len(dataset), args.interval))
    if args.pred_dir:
        infos = getattr(dataset, 'data_infos', None)
        if infos is None or len(infos) != len(dataset):
            raise ValueError('Offline prediction visualization requires dataset.data_infos for token lookup')
        selected_indices = [i for i in selected_indices
                            if infos[i]['token'] in prediction_files]
    print(f'Visualizing {len(selected_indices)}/{len(dataset)} frames (interval={args.interval})')
    selected_dataset = torch.utils.data.Subset(dataset, selected_indices)
    dataflow = build_dataloader(
        selected_dataset,
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

    for data in tqdm(dataflow):
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

        #* (11.12) OCC 图片与原始数组分目录，预测和 GT 合为一幅俯视图。
        occ_pred = outputs[0].get('occ_pred') if args.mode == 'pred' else None
        occ_gt = data['voxel_semantics'][0] if 'voxel_semantics' in data else None
        if occ_pred is not None:
            save_occupancy_array(
                os.path.join(frame_out_dir, 'occ', 'npy', f'{name}.npy'), occ_pred)
        if occ_gt is not None:
            save_occupancy_array(
                os.path.join(frame_out_dir, 'occ', 'npy', f'{name}_gt.npy'), occ_gt)
        if occ_pred is not None or occ_gt is not None:
            render_occupancy_comparison(
                os.path.join(frame_out_dir, 'occ', 'images', f'{name}.png'),
                occ_pred, occ_gt, name)

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
