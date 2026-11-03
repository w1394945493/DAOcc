#!/usr/bin/env python3
"""三模态对照：每行依次显示最终共享BEV、Occ预测和GT，按场景保存。

Occ使用最高有效占据体素的BEV投影，不是三维透视渲染。
默认在各模型Occ预测图叠加对应3D检测框的俯视轮廓和朝向，不修改GT图。
三个配置应具有相同的验证pipeline；数据只加载一次，保证比较同一输入。
"""
import argparse
import ast
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_occ_colors(path):
    # * 直接读取指定 vis_occ3d.py 中 colors 的原始列表，不另定义类别配色。
    # 只解析赋值表达式，不导入整个脚本，避免执行其 Qt 环境变量设置。
    import numpy as np
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "colors" for t in node.targets):
            # 当前源文件格式：colors = np.array([...]).astype(np.uint8)。
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "astype":
                value = value.func.value
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "array"):
                break
            palette = np.asarray(ast.literal_eval(value.args[0]), dtype=np.uint8)
            if palette.ndim != 2 or palette.shape[0] < 17 or palette.shape[1] < 3:
                break
            return palette[:, :3]
    raise ValueError(f"无法读取 {path} 的 colors 配色表；请核对源文件格式")

def occupancy_rgb(semantics, palette, mask=None):
    import numpy as np
    labels = top_labels(semantics, mask).T
    # * 空体素单独留白：vis_occ3d.py 的第 17 项是 ego car 装饰色，不能当成 free 类颜色。
    rgb = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    visible = (labels >= 0) & (labels < 17)
    rgb[visible] = palette[labels[visible].astype(int)]
    return rgb

def select_indices(total, requested):
    # * 在时间排序的数据集索引上均匀取样；超过总数时取全部，不循环、不重复。
    if total < 1 or requested < 1:
        raise ValueError("数据集样本数和 --num-samples 都必须大于 0")
    count = min(total, requested)
    if count == 1:
        return [total // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]

def clear_pretrained(value):
    # checkpoint 已包含权重，不另外读取 backbone 的预训练文件。
    if isinstance(value, dict):
        if "pretrained" in value:
            value["pretrained"] = None
        init = value.get("init_cfg")
        if isinstance(init, dict) and init.get("type") == "Pretrained":
            value["init_cfg"] = None
        for child in value.values():
            clear_pretrained(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            clear_pretrained(child)

def top_labels(semantics, camera_mask=None):
    import numpy as np
    # * Occ3D 数组顺序为 [X, Y, Z]；每个 XY 柱取最高的有效占据体素。
    # 17 为自由空间，255 为忽略；类别 0 仍是有效占据类别。
    valid = (semantics >= 0) & (semantics < 17)
    if camera_mask is not None:
        valid &= camera_mask.astype(bool)
    heights = np.where(valid, np.arange(semantics.shape[2]), -1).max(axis=2)
    labels = np.take_along_axis(semantics, np.maximum(heights, 0)[..., None], axis=2)[..., 0]
    return np.where(heights >= 0, labels, 17)


def boxes_to_occ(boxes, scores, labels, data, score_thr):
    """将预测3D框的8角点从增强LiDAR坐标转换到Occ自车坐标。"""
    import numpy as np
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    scores, labels = array(scores), array(labels)
    values = array(boxes.tensor)
    keep = np.flatnonzero(np.isfinite(scores) & (scores >= score_thr)
                         & np.isfinite(values[:, :7]).all(1)
                         & (values[:, 3:6] > 0).all(1))
    if not len(keep):
        return []
    # corners在空框时不支持，因此先检查keep；按分数降序绘制。
    keep = keep[np.argsort(-scores[keep])]
    corners = array(boxes.corners)[keep].astype(np.float64)
    aug = array(data["lidar_aug_matrix"])[0]
    l2e = array(data["lidar2ego"])[0]
    occ = array(data["occ_aug_matrix"])[0]
    #* 与CrossCoordinateSample查询变换互为逆向：撤销LiDAR增强→自车→Occ增强。
    # Occ采样器仅使用occ_aug_matrix的3×3部分，这里保持一致，不应用其平移。
    corners = (corners - aug[:3, 3]) @ np.linalg.inv(aug[:3, :3]).T
    corners = corners @ l2e[:3, :3].T + l2e[:3, 3]
    corners = corners @ occ[:3, :3].T
    return [dict(corners=c, score=float(scores[k]), label=int(labels[k]))
            for c, k in zip(corners, keep)]


def draw_detection_boxes(ax, detections, class_names, show_labels=False):
    """绘制完整3D框投影的凸包与朝向；固定显示范围，框线在边界自动裁剪。"""
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError
    for detection in detections:
        corners = detection["corners"]
        if not np.isfinite(corners).all():
            continue
        xy = corners[:, :2]
        if (xy[:, 0].max() < -40 or xy[:, 0].min() > 40
                or xy[:, 1].max() < -40 or xy[:, 1].min() > 40):
            continue
        #* 投影8角点的凸包，保留LiDAR到自车变换中俯仰/横滚造成的投影变化。
        try:
            hull = ConvexHull(xy)
        except QhullError:
            continue
        outline = xy[np.r_[hull.vertices, hull.vertices[0]]]
        ax.plot(outline[:, 0], outline[:, 1], color="black", lw=.8, zorder=5, clip_on=True)
        center = xy.mean(0)
        front = xy[[4, 5, 6, 7]].mean(0)  # LiDAR框局部+x方向为朝向
        ax.annotate("", xy=front, xytext=center,
                    arrowprops=dict(arrowstyle="->", color="black", lw=.65),
                    annotation_clip=True, zorder=6)
        if show_labels:
            label = detection["label"]
            name = class_names[label] if 0 <= label < len(class_names) else str(label)
            ax.text(center[0], center[1], f"{name} {detection['score']:.2f}",
                    fontsize=4, color="black", clip_on=True, zorder=7,
                    bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=.2))
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)



def save_scene_videos(frames, output_dir, fps):
    """仅使用本次清单中的图片，每个场景独立编码，避免混入旧运行结果。"""
    import cv2
    import numpy as np
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = {}
    for frame in frames:
        scenes.setdefault((frame["scene"], frame["scene_token"]), []).append(frame)
    videos = []
    for (scene, scene_token), scene_frames in scenes.items():
        #* 按原始关键帧序号排序，不按图片文件时间或推理完成顺序排序。
        scene_frames.sort(key=lambda f: f["scene_frame_number"])
        first = cv2.imread(scene_frames[0]["path"])
        if first is None:
            raise RuntimeError(f"无法读取视频帧：{scene_frames[0]['path']}")
        height, width = first.shape[:2]
        # mp4编码使用偶数尺寸；右侧/下侧不足一个像素时补白，不裁剪内容。
        width, height = width + width % 2, height + height % 2
        path = output_dir / f"{scene}_{scene_token}.mp4"
        temporary = output_dir / f"{scene}_{scene_token}.partial.mp4"
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            temporary.unlink(missing_ok=True)
            raise RuntimeError("OpenCV无法创建MP4视频，请检查本机视频编码支持和输出目录权限")
        try:
            for frame in scene_frames:
                image = cv2.imread(frame["path"])
                if image is None:
                    raise RuntimeError(f"无法读取视频帧：{frame['path']}")
                h, w = image.shape[:2]
                #* tight裁边可能使图片尺寸略有差异：保持纵横比缩放，居中补白。
                ratio = min(width / w, height / h, 1.0)
                if ratio < 1:
                    image = cv2.resize(image, (max(1, round(w*ratio)), max(1, round(h*ratio))),
                                       interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]
                canvas = np.full((height, width, 3), 255, dtype=np.uint8)
                y, x = (height-h)//2, (width-w)//2
                canvas[y:y+h, x:x+w] = image
                writer.write(canvas)
        except Exception:
            writer.release()
            temporary.unlink(missing_ok=True)
            raise
        writer.release()
        temporary.replace(path)
        videos.append(dict(scene=scene, scene_token=scene_token, path=str(path.resolve()),
                           frames=len(scene_frames), fps=fps, size=[width, height],
                           frame_numbers=[f["scene_frame_number"] for f in scene_frames]))
        print(f"视频：{path}（{len(scene_frames)}帧，{fps:g} FPS）", flush=True)
    return videos


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("camera", "lidar", "fusion"):
        parser.add_argument(f"--{name}-config", required=True)
        parser.add_argument(f"--{name}-checkpoint", required=True,
                            help="明确指定普通或EMA权重；不自动替换checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20,
                        help="均匀采样帧数；指定--num-scenes时表示每个场景的帧数")
    parser.add_argument("--num-scenes", type=int,
                        help="连续选择M个验证场景，从--scene-index开始（默认从0开始）")
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument("--scene", help="可选场景名或scene token；在该场景内均匀抽样")
    scene_group.add_argument("--scene-index", type=int,
                             help="验证集场景索引，从0开始，按场景在data_infos中首次出现的顺序")
    parser.add_argument("--list-scenes", action="store_true",
                        help="列出验证集场景索引、名称和帧数后退出，不加载模型权重")
    parser.add_argument("--all-frames", action="store_true", help="保存所选范围内所有帧")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--camera-mask", action="store_true",
                        help="三个预测和GT统一应用mask_camera，与掩码评估范围对应")
    parser.add_argument("--feature-scale", choices=("independent", "shared"),
                        default="independent", help="独立色标看结构；共享色标比较响应幅值")
    parser.add_argument("--no-boxes", action="store_true", help="关闭Occ预测图上的检测框")
    parser.add_argument("--box-score-thr", type=float, default=0.25, help="检测框显示阈值，默认0.25")
    parser.add_argument("--box-labels", action="store_true", help="同时标注检测类别和置信度")
    parser.add_argument("--video-dir", type=Path, help="可选：每个场景生成一个MP4，统一保存到此目录")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="视频帧率，默认2；抽样帧等间隔播放，不还原缺失帧")
    args = parser.parse_args()
    import math
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("video-fps必须为有限正数")
    if not 0 <= args.box_score_thr <= 1:
        parser.error("box-score-thr必须在[0,1]内")
    if args.num_samples < 1 or args.workers < 0:
        parser.error("num-samples必须大于0，workers不能为负")
    if args.scene_index is not None and args.scene_index < 0:
        parser.error("scene-index必须大于等于0")
    if args.num_scenes is not None:
        if args.num_scenes < 1:
            parser.error("num-scenes必须大于0")
        if args.scene is not None:
            parser.error("num-scenes请配合scene-index使用，不能与scene同时指定")
    return args


def scene_lookup(dataset):
    #* 沿nuScenes关键帧链表编号，同场景下保留原帧序号，不按抽样顺序重新编号。
    table = Path(dataset.dataset_root) / dataset.version
    scenes = json.loads((table / "scene.json").read_text())
    samples = {s["token"]: s for s in json.loads((table / "sample.json").read_text())}
    lookup = {}
    for scene in scenes:
        token, number = scene["first_sample_token"], 1
        while token:
            if token in lookup or samples[token]["scene_token"] != scene["token"]:
                raise ValueError(f"场景链表异常：{token}")
            lookup[token] = (scene["name"], scene["token"], number)
            token, number = samples[token]["next"], number + 1
    return lookup


def first_tensor(value):
    while isinstance(value, (list, tuple)):
        value = value[0]
    return value


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
    from torchpack.utils.config import Config as TorchpackConfig
    from torch.utils.data import Subset
    from tqdm import tqdm
    from mmdet.apis import set_random_seed
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model

    set_random_seed(0, deterministic=True)
    device = torch.device("cuda", args.gpu)
    names = ("camera", "lidar", "fusion")
    titles = ("Camera only", "LiDAR only", "Camera + LiDAR")
    configs, models, samplers = [], [], []
    for name in names:
        #* 各模型独立读取配置，避免递归合并时残留另一模态的encoder设置。
        raw = TorchpackConfig()
        raw.load(str(Path(getattr(args, name + "_config")).resolve()), recursive=True)
        configs.append(Config(recursive_eval(raw)))
    # 数据只从融合配置构建；拒绝不一致的验证配置，防止误用图像归一化或增强。
    if any(c.data.val != configs[2].data.val for c in configs[:2]):
        raise ValueError("三模型data.val不一致，请先统一验证集和预处理pipeline")
    data_cfg = deepcopy(configs[2].data.val)
    data_cfg.test_mode = True
    data_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(data_cfg)
    lookup = scene_lookup(dataset)
    #* 只统计当前验证集包含的场景，不使用nuScenes全量scene.json的索引。
    # 字典保留插入顺序，因此索引由当前验证集data_infos中的首次出现顺序决定。
    scenes = {}
    for info in dataset.data_infos:
        scene_name, scene_token, _ = lookup[info["token"]]
        if scene_token not in scenes:
            scenes[scene_token] = dict(name=scene_name, token=scene_token, frames=0)
        scenes[scene_token]["frames"] += 1
    scene_list = list(scenes.values())
    if args.list_scenes:
        print("验证集场景索引（从0开始） | 场景名 | 验证帧数 | scene token")
        for i, scene in enumerate(scene_list):
            print(f"{i:3d} | {scene['name']} | {scene['frames']} | {scene['token']}")
        return
    if args.scene_index is not None and args.num_scenes is None:
        if args.scene_index >= len(scene_list):
            raise ValueError(f"scene-index={args.scene_index}越界；验证集共{len(scene_list)}个场景")
        selected_scene = scene_list[args.scene_index]
        args.scene = selected_scene["token"]
        print(f"选择验证集场景[{args.scene_index}] {selected_scene['name']}，"
              f"共{selected_scene['frames']}帧", flush=True)
    candidates = [i for i, info in enumerate(dataset.data_infos)
                  if args.scene is None or args.scene in lookup[info["token"]][:2]]
    if not candidates:
        raise ValueError(f"没有找到所选场景的验证帧：{args.scene}")
    indices = candidates if args.all_frames else [candidates[i] for i in
               select_indices(len(candidates), args.num_samples)]
    if args.num_scenes is not None:
        #* 多场景模式：先选连续M个场景，再分别在各场景内均匀抽N帧。
        # 不能把M个场景混在一起抽N帧，否则无法保证每个场景都有相同样本数。
        start = args.scene_index if args.scene_index is not None else 0
        stop = start + args.num_scenes
        if stop > len(scene_list):
            raise ValueError(f"请求场景索引[{start},{stop})，但验证集只有{len(scene_list)}个场景")
        indices = []
        for scene_index in range(start, stop):
            scene = scene_list[scene_index]
            frame_indices = [i for i, info in enumerate(dataset.data_infos)
                             if lookup[info["token"]][1] == scene["token"]]
            # 按场景内原始关键帧序号排序，确保均匀采样沿时间顺序进行。
            frame_indices.sort(key=lambda i: lookup[dataset.data_infos[i]["token"]][2])
            chosen = frame_indices if args.all_frames else [frame_indices[j] for j in
                      select_indices(len(frame_indices), args.num_samples)]
            indices.extend(chosen)
            print(f"场景[{scene_index}] {scene['name']}：{len(frame_indices)}帧中选择{len(chosen)}帧",
                  flush=True)
    torch.cuda.set_device(args.gpu)
    loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                             workers_per_gpu=args.workers, dist=False, shuffle=False)
    for name, cfg in zip(names, configs):
        model_cfg = deepcopy(cfg.model)
        clear_pretrained(model_cfg)
        print(f"加载{name}模型和指定checkpoint……", flush=True)
        previous = logging.root.manager.disable
        try:
            logging.disable(max(previous, logging.INFO))
            model = build_model(model_cfg)
        finally:
            logging.disable(previous)
        #* 严格加载完整权重；不允许单模态结构不匹配时悄悄使用随机初始化层。
        load_checkpoint(model, str(Path(getattr(args, name + "_checkpoint")).expanduser()),
                        map_location="cpu", strict=True)
        model.float().eval()
        for module in model.modules():
            if hasattr(module, "fp16_enabled"):
                module.fp16_enabled = False
        transform = cfg.model.heads.occ.coordinate_transform
        if (transform.type != "CrossCoordinateSample" or
                list(transform.point_num) != [200, 200, 1] or
                list(transform.point_range) != [-40., 40., -40., 40., 0., 0.]):
            raise ValueError("仅支持Occ3D自车系±40m、200×200的坐标变换")
        sampler = deepcopy(model.heads["occ"].coordinate_transform)
        if sampler.point_type != "ego":
            raise ValueError("特征必须对齐到ego坐标系")
        # 只复用坐标变换，不使用Occ head的额外学习通道映射。
        sampler.transfer_conv, sampler.extra_up = None, False
        models.append(model)
        samplers.append(sampler.float().eval())

    colors = load_occ_colors(ROOT / "tools/vis_occ3d.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(models={n: dict(config=str(Path(getattr(args,n+"_config")).resolve()),
                                  checkpoint=str(Path(getattr(args,n+"_checkpoint")).expanduser().resolve()))
                          for n in names}, selected_indices=indices, precision="fp32",
                  scene_index=args.scene_index, num_scenes=args.num_scenes,
                  num_samples=args.num_samples, all_frames=args.all_frames,
                  feature_layer="decoder.neck", feature_scale=args.feature_scale,
                  feature_reduction="mean(abs(channels)) after ego geometric resampling",
                  occupancy="highest occupied voxel per XY column, labels 0..16; free=17",
                  coordinates="ego XY [-40,40] metres; horizontal X, vertical Y",
                  detection_boxes=not args.no_boxes, box_score_thr=args.box_score_thr,
                  box_labels=args.box_labels,
                  box_coordinates="inverse lidar augmentation -> lidar2ego -> occ linear augmentation",
                  box_mask="not clipped by mask_camera; clipped to plot extent",
                  camera_mask=args.camera_mask, layout="3 rows: camera/lidar/fusion; columns: BEV/prediction/GT",
                  frames=[])
    with torch.no_grad():
        for index, batch in zip(indices, tqdm(loader, total=len(indices), desc="三模型对照")):
            data = scatter(batch, [args.gpu])[0]
            sem = first_tensor(data["voxel_semantics"])
            if sem.ndim == 4:
                sem = sem[0]
            sem = sem.cpu().numpy()
            if sem.shape != (200, 200, 16):
                raise ValueError(f"非Occ3D GT形状：{sem.shape}")
            mask = None
            if args.camera_mask:
                mask = first_tensor(data["mask_camera"])
                if mask.ndim == 4:
                    mask = mask[0]
                mask = mask.cpu().numpy()
            gt_rgb = occupancy_rgb(sem, colors, mask)
            heats, predictions, stats, detections, detection_classes = [], [], [], [], []
            for name, model, sampler in zip(names, models, samplers):
                captured = {}
                def capture(module, inputs, output):
                    if isinstance(output, (list, tuple)):
                        if len(output) != 1:
                            raise ValueError("decoder.neck应输出单个BEV张量")
                        output = output[0]
                    captured["feature"] = output.detach()
                #* 只让一个模型驻留GPU；每帧的三个模型使用同一份scatter后的输入。
                model.to(device)
                sampler.to(device)
                handle = model.decoder["neck"].register_forward_hook(capture)
                try:
                    outputs = model(**data)
                    feature = captured.pop("feature").float()
                    aligned = sampler(feature.transpose(-1, -2).contiguous(),
                                      data["lidar_aug_matrix"].float(), data["lidar2ego"].float(),
                                      data["occ_aug_matrix"].float())
                    heat = aligned[0].abs().mean(0).cpu().numpy()
                    if not np.isfinite(heat).all():
                        raise ValueError(f"{name} BEV特征包含NaN/Inf")
                    prediction = outputs[0]["occ_pred"]
                    if torch.is_tensor(prediction):
                        prediction = prediction.detach().cpu().numpy()
                    prediction = np.asarray(prediction)
                    if prediction.shape != sem.shape:
                        raise ValueError(f"{name}预测与GT形状不一致")
                    heats.append(heat)
                    predictions.append(occupancy_rgb(prediction, colors, mask))
                    #* 检测结果已在同次推理中生成，无需额外运行模型。
                    # camera-mask只作用于Occ体素，不删除框；框按分数筛选并裁剪到显示范围。
                    boxes = []
                    classes = []
                    if not args.no_boxes:
                        if not all(k in outputs[0] for k in ("boxes_3d", "scores_3d", "labels_3d")):
                            raise ValueError(f"{name}未输出检测框；请启用object head或使用--no-boxes")
                        boxes = boxes_to_occ(outputs[0]["boxes_3d"], outputs[0]["scores_3d"],
                                             outputs[0]["labels_3d"], data, args.box_score_thr)
                        classes = [c for task in model.heads["object"].class_names for c in task]
                    detections.append(boxes)
                    detection_classes.append(classes)
                    stats.append(dict(model=name, feature_shape=list(feature.shape),
                                      heat_mean=float(heat.mean()), heat_max=float(heat.max()),
                                      boxes_after_score_filter=len(boxes)))
                    del outputs, feature, aligned, prediction
                finally:
                    handle.remove()
                    captured.clear()
                    sampler.cpu()
                    model.cpu()
                torch.cuda.empty_cache()
            # 默认每模型独立百分位归一化，仅比较结构；共享模式使用三图联合色标。
            shared = np.percentile(np.stack(heats), [1,99])
            fig, axes = plt.subplots(3, 3, figsize=(10,10))
            for row, (title, heat, pred) in enumerate(zip(titles, heats, predictions)):
                low, high = shared if args.feature_scale == "shared" else np.percentile(heat,[1,99])
                normalized = np.clip((heat-low)/max(float(high-low),1e-12),0,1)
                stats[row].update(heat_p01=float(low), heat_p99=float(high))
                axes[row,0].imshow(normalized, origin="lower", extent=(-40,40,-40,40),
                                   cmap="jet", vmin=0, vmax=1, interpolation="nearest")
                axes[row,1].imshow(pred, origin="lower", extent=(-40,40,-40,40), interpolation="nearest")
                if not args.no_boxes:
                    draw_detection_boxes(axes[row,1], detections[row], detection_classes[row], args.box_labels)
                axes[row,2].imshow(gt_rgb, origin="lower", extent=(-40,40,-40,40), interpolation="nearest")
                for col, label in enumerate((title+" — BEV", title+(" — Occ" if args.no_boxes else " — Occ + boxes"), "Occ GT")):
                    ax = axes[row,col]
                    ax.set_aspect("equal")
                    ax.set_axis_off()
                    ax.set_title(label, fontsize=10, pad=3)
            fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.975,wspace=.015,hspace=.09)
            token = dataset.data_infos[index]["token"]
            scene, scene_token, number = lookup[token]
            path = args.output_dir / scene / f"frame_{number:03d}_{token}.png"
            path.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(path,dpi=180,bbox_inches="tight",pad_inches=.02,facecolor="white")
            plt.close(fig)
            report["frames"].append(dict(index=index,token=token,scene=scene,scene_token=scene_token,
                                         scene_frame_number=number,path=str(path.resolve()),models=stats))
            (args.output_dir/"comparison_manifest.json").write_text(
                json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            del data, batch
    #* 可选视频仅编码已保存的PNG，不再次推理，也不把所有帧同时读入内存。
    if args.video_dir is not None:
        report["videos"] = save_scene_videos(report["frames"], args.video_dir, args.video_fps)
        (args.output_dir / "comparison_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"已保存{len(report['frames'])}帧对比图：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
