#!/usr/bin/env python3
"""均匀抽取真实训练样本，统计相机分支各阶段耗时并可选可视化。

本脚本复现正式训练中的相机分支：六路增强图像依次经过camera backbone、
camera neck和视角变换，最终得到用于BEV融合的图像BEV特征。脚本不会执行
LiDAR体素化、LiDAR backbone、跨模态融合、decoder或任务head。

可视化左侧保持六路训练输入图像的2x3布局，右侧放置图像BEV特征热力图；
两个部分在同一行，六路图像之间只保留很小的间隔。
训练pipeline仍加载聚合LiDAR，以保持数据处理和正式训练一致，但不再绘制点云。

示例：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python \
      tools/visualize_camera_bev_pipeline.py \
      configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_fixed_s128.yaml \
      --gpu 0 --num-samples 20 --warmup-iters 3 --history-sweeps 9 \
      --checkpoint /path/to/epoch_1.pth --visualize \
      --output-dir /tmp/camera_bev_visualization
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from torchpack.utils.config import Config as TorchpackConfig

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_backbone, build_neck
from mmdet3d.models.builder import build_vtransform
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="训练YAML配置")
    parser.add_argument("--num-samples", type=int, default=10, help="均匀抽取的样本数")
    parser.add_argument("--warmup-iters", type=int, default=1, help="正式计时前的预热次数")
    parser.add_argument(
        "--history-sweeps", type=int,
        help="覆盖训练pipeline中的历史点云帧数；仅影响LiDAR输入图",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="可选训练权重；提供后加载encoders.camera下的真实模型参数",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="保存根目录")
    parser.add_argument("--visualize", action="store_true", help="保存每个样本的组合可视化")
    parser.add_argument("--max-plot-points", type=int, default=60000)
    args, overrides = parser.parse_known_args()
    if args.num_samples <= 0:
        parser.error("--num-samples必须为正整数")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters必须为非负整数")
    if args.history_sweeps is not None and args.history_sweeps < 0:
        parser.error("--history-sweeps必须为非负整数")
    if args.max_plot_points <= 0:
        parser.error("--max-plot-points必须为正整数")
    return args, overrides


def load_config(path, overrides):
    raw = TorchpackConfig()
    raw.load(path, recursive=True)
    raw.update(overrides)
    return Config(recursive_eval(raw), filename=path)


def inner_dataset_cfg(train_cfg):
    current = train_cfg
    while "pipeline" not in current:
        if "dataset" not in current:
            raise KeyError("data.train中没有找到包含pipeline的内层数据集")
        current = current["dataset"]
    return current


def configure_history_sweeps(train_cfg, override):
    pipeline = inner_dataset_cfg(train_cfg)["pipeline"]
    matches = [step for step in pipeline if step.get("type") == "LoadPointsFromMultiSweeps"]
    if len(matches) != 1:
        raise RuntimeError("训练pipeline应恰好包含一个LoadPointsFromMultiSweeps")
    if override is not None:
        matches[0]["sweeps_num"] = override
    return int(matches[0]["sweeps_num"])


def unwrap_data_container(value):
    while value.__class__.__name__ == "DataContainer":
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        while value.__class__.__name__ == "DataContainer":
            value = value.data
    return value


def unwrap_dataset_for_uniform_sampling(dataset):
    """去掉CBGS/Repeat等单数据集包装，在原始样本范围内均匀抽样。"""
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def uniform_dataset_indices(dataset_length, num_samples):
    if num_samples > dataset_length:
        raise ValueError(f"num-samples={num_samples}超过数据集长度{dataset_length}")
    return np.linspace(0, dataset_length - 1, num_samples, dtype=np.int64).tolist()


def tensor_to_device(value, device, add_batch=True):
    value = unwrap_data_container(value)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.contiguous().to(device)
    if add_batch:
        value = value.unsqueeze(0)
    return value


def load_training_sample(dataset, dataset_index, history_sweeps, args, device):
    """通过正式训练pipeline取得增强图像、聚合点云和投影矩阵。"""
    annotation_info = dataset.data_infos[dataset_index]
    raw_info = dataset.get_data_info(dataset_index)
    sample = dataset[dataset_index]
    if sample is None:
        raise RuntimeError(f"dataset[{dataset_index}]返回None")

    #* 图像保持训练pipeline输出：[6,3,H,W]，增加batch维后送入相机分支。
    images = tensor_to_device(sample["img"], device, add_batch=True).float()
    points = unwrap_data_container(sample["points"])
    if hasattr(points, "tensor"):
        points = points.tensor
    points = points.contiguous().float().to(device)

    matrix_names = (
        "camera2ego", "lidar2ego", "lidar2camera", "lidar2image",
        "camera_intrinsics", "camera2lidar", "img_aug_matrix", "lidar_aug_matrix",
    )
    matrices = {name: tensor_to_device(sample[name], device).float() for name in matrix_names}
    metas = unwrap_data_container(sample["metas"])

    # 相机顺序来自当前样本info中的cams插入顺序，与pipeline读取顺序完全一致。
    camera_names = list(annotation_info.get("cams", {}).keys())
    if len(camera_names) != images.shape[1]:
        camera_names = [f"camera_{index}" for index in range(images.shape[1])]

    source = {
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_original_index": dataset_index,
        "scene_token": str(annotation_info.get("scene_token", "")),
        "frame_token": str(annotation_info.get("token", raw_info.get("sample_idx", ""))),
        "camera_names": camera_names,
        "configured_history_sweeps": history_sweeps,
        "available_history_sweeps": len(raw_info.get("sweeps", [])),
    }
    return images, points, matrices, metas, source


def initialize_module(module):
    if hasattr(module, "init_weights"):
        module.init_weights()


def load_camera_checkpoint(modules, checkpoint_path):
    """只提取checkpoint中encoders.camera下的参数，不构建完整多模态模型。"""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefixes = ("module.encoders.camera.", "encoders.camera.")
    extracted = {}
    for name, value in state_dict.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                extracted[name[len(prefix):]] = value
                break
    if not extracted:
        raise KeyError("checkpoint中未找到encoders.camera参数")

    incompatible = modules.load_state_dict(extracted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "相机分支权重不匹配：missing=%s, unexpected=%s"
            % (incompatible.missing_keys, incompatible.unexpected_keys)
        )


def build_camera_modules(cfg, device, checkpoint_path=None):
    camera_cfg = cfg.model.encoders.camera
    modules = torch.nn.ModuleDict({
        "backbone": build_backbone(camera_cfg.backbone),
        "neck": build_neck(camera_cfg.neck),
        "vtransform": build_vtransform(camera_cfg.vtransform),
    })
    if camera_cfg.get("bev_backbone") is not None:
        modules["bev_backbone"] = build_backbone(camera_cfg.bev_backbone)
    if camera_cfg.get("bev_neck") is not None:
        modules["bev_neck"] = build_neck(camera_cfg.bev_neck)

    for module in modules.values():
        initialize_module(module)
    modules = modules.to(device).eval()

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint不存在：{checkpoint_path}")
        load_camera_checkpoint(modules, checkpoint_path)
    return modules


@torch.no_grad()
def run_camera_branch(images, points, matrices, metas, modules):
    """复现正式相机分支，并用CUDA Event统计各阶段GPU耗时。"""
    has_bev_backbone = "bev_backbone" in modules
    has_bev_neck = "bev_neck" in modules
    stage_names = ["camera_backbone_ms", "camera_neck_ms", "view_transform_ms"]
    if has_bev_backbone:
        stage_names.append("camera_bev_backbone_ms")
    if has_bev_neck:
        stage_names.append("camera_bev_neck_ms")
    events = [torch.cuda.Event(enable_timing=True) for _ in range(len(stage_names) + 1)]

    batch_size, num_cameras, channels, height, width = images.shape
    x = images.reshape(batch_size * num_cameras, channels, height, width)
    event_index = 0
    events[event_index].record()

    #* ==================== 1. 图像backbone =======================
    # 将6路图像合并到batch维，一次经过ResNet，输出多尺度透视图特征。
    x = modules["backbone"](x)
    event_index += 1
    events[event_index].record()

    #* ==================== 2. 图像neck ============================
    # GeneralizedLSSFPN融合ResNet多层特征，得到统一通道的图像平面特征。
    x = modules["neck"](x)
    if not isinstance(x, torch.Tensor):
        x = x[0]
    neck_feature_shape = list(x.shape)
    _, feature_channels, feature_height, feature_width = x.shape
    x = x.reshape(
        batch_size, num_cameras, feature_channels, feature_height, feature_width
    )
    event_index += 1
    events[event_index].record()

    #* ==================== 3. 图像到BEV视角变换 ====================
    # 使用相机内外参、图像增强矩阵和LiDAR增强矩阵，将六路透视特征采样到
    # 统一三维网格；最终把高度层折叠到通道维，生成图像BEV特征。
    x = modules["vtransform"](
        x,
        [points],
        matrices["camera2ego"],
        matrices["lidar2ego"],
        matrices["lidar2camera"],
        matrices["lidar2image"],
        matrices["camera_intrinsics"],
        matrices["camera2lidar"],
        matrices["img_aug_matrix"],
        matrices["lidar_aug_matrix"],
        [metas],
    )
    event_index += 1
    events[event_index].record()

    #* ==================== 4. 配置中的可选相机BEV后处理 ====================
    # 当前DAOcc配置通常没有这两层；若后续配置增加，脚本仍按正式路径执行和计时。
    if has_bev_backbone:
        x = modules["bev_backbone"](x)
        event_index += 1
        events[event_index].record()
    if has_bev_neck:
        x = modules["bev_neck"](x)
        event_index += 1
        events[event_index].record()

    events[-1].synchronize()
    timings = {
        name: float(events[index].elapsed_time(events[index + 1]))
        for index, name in enumerate(stage_names)
    }
    timings["total_ms"] = float(events[0].elapsed_time(events[-1]))
    return {
        "bev_features": x,
        "neck_feature_shape": neck_feature_shape,
        "timings_ms": timings,
    }


def uniform_subsample(array, limit, seed):
    if len(array) <= limit:
        return array
    rng = np.random.default_rng(seed)
    return array[rng.choice(len(array), limit, replace=False)]


def denormalize_images(images, metas):
    """把训练pipeline归一化后的RGB张量恢复到[0,1]，仅供绘图。"""
    norm = metas.get("img_norm_cfg", {}) if isinstance(metas, dict) else {}
    mean = torch.tensor(norm.get("mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1)
    std = torch.tensor(norm.get("std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1)
    restored = images[0].detach().float().cpu() * std + mean
    return restored.clamp(0, 1).permute(0, 2, 3, 1).numpy()


def save_visualization(images, metas, source, outputs, cfg, path):
    restored_images = denormalize_images(images, metas)

    # 对全部通道取绝对值均值，得到每个BEV网格的整体激活强度。
    # 该热力图只用于观察特征空间分布，不参与模型计算。
    bev = outputs["bev_features"][0].detach().float().cpu()
    bev_heatmap = bev.abs().mean(dim=0).numpy()
    point_range = list(cfg.get("point_cloud_range", [-54, -54, -5, 54, 54, 3]))

    #* 左侧六路图像保持2x3布局，右侧BEV与整个图像块横向并列。
    # 内层image_grid只保留极小的水平/垂直间隔；outer的间隔用于区分
    # 环视图像块和BEV热力图，避免把六路图像误改成1x6排列。
    fig = plt.figure(figsize=(20, 3.9), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[4.05, 1.0], wspace=0.008)
    image_grid = outer[0, 0].subgridspec(2, 3, wspace=0.002, hspace=0.004)
    camera_names = source["camera_names"]
    for index in range(len(restored_images)):
        row, column = divmod(index, 3)
        axis = fig.add_subplot(image_grid[row, column])
        # aspect='auto'让图像填满分配到的网格单元，避免imshow为保持原始
        # 宽高比而在相邻图像之间留下看似很大的空白列。
        axis.imshow(restored_images[index], aspect="auto")
        # 相机名称直接叠加在图像内部，避免标题单独占据一行白色空间。
        axis.text(
            0.5,
            0.985,
            camera_names[index],
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", alpha=0.48, edgecolor="none", pad=1.2),
        )
        axis.axis("off")

    bev_axis = fig.add_subplot(outer[0, 1])
    heatmap = bev_axis.imshow(
        bev_heatmap,
        origin="lower",
        extent=[point_range[0], point_range[3], point_range[1], point_range[4]],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    bev_axis.set_title(
        f"Camera BEV |feature| mean ({bev.shape[0]} channels)", fontsize=9, pad=3
    )
    # 可视化只观察BEV激活分布，不显示X/Y刻度和轴标题，以去掉左侧及下侧留白。
    bev_axis.set_xticks([])
    bev_axis.set_yticks([])
    bev_axis.set_xlabel("")
    bev_axis.set_ylabel("")
    bev_axis.grid(False)
    colorbar = fig.colorbar(
        heatmap, ax=bev_axis, fraction=0.046, pad=0.025, label="mean |activation|"
    )
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("mean |activation|", fontsize=8)
    fig.subplots_adjust(left=0.001, right=0.998, bottom=0.002, top=0.985)
    # bbox_inches='tight'裁掉画布外侧未使用区域；极小pad只防止坐标文字被截断。
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)


def summarize_timings(sample_results):
    timing_keys = list(sample_results[0]["timings_ms"])
    summary = {}
    for key in timing_keys:
        values = np.asarray([item["timings_ms"][key] for item in sample_results])
        summary[key] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main():
    args, overrides = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("相机BEV测速需要CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config, overrides)
    history_sweeps = configure_history_sweeps(cfg.data.train, args.history_sweeps)
    wrapped_dataset = build_dataset(cfg.data.train)
    dataset = unwrap_dataset_for_uniform_sampling(wrapped_dataset)
    selected_indices = uniform_dataset_indices(len(dataset), args.num_samples)
    modules = build_camera_modules(cfg, device, args.checkpoint)
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 首个样本用于CUDA/cuDNN预热，预热结果不纳入正式统计。
    if args.warmup_iters:
        images, points, matrices, metas, _ = load_training_sample(
            dataset, selected_indices[0], history_sweeps, args, device
        )
        for _ in range(args.warmup_iters):
            warmup_outputs = run_camera_branch(images, points, matrices, metas, modules)
            del warmup_outputs
        del images, points, matrices, metas
        torch.cuda.empty_cache()

    sample_results = []
    for order, dataset_index in enumerate(selected_indices):
        images, points, matrices, metas, source = load_training_sample(
            dataset, dataset_index, history_sweeps, args, device
        )
        outputs = run_camera_branch(images, points, matrices, metas, modules)

        scene_token = source["scene_token"] or "scene_unknown"
        frame_token = source["frame_token"] or f"frame_index_{dataset_index}"
        sample_dir = root / f"{scene_token}_{frame_token}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        time_values, time_counts = torch.unique(points[:, 4].cpu(), sorted=True, return_counts=True)
        result = {
            "sample_order": order,
            "source": {**source, "output_directory": str(sample_dir)},
            "camera_checkpoint": (
                str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None
            ),
            "input_image_shape": list(images.shape),
            "num_input_points_for_visualization": len(points),
            "camera_neck_feature_shape": outputs["neck_feature_shape"],
            "camera_bev_feature_shape": list(outputs["bev_features"].shape),
            "timings_ms": outputs["timings_ms"],
            "time_lag_groups": [
                {"seconds": float(value), "num_points": int(count)}
                for value, count in zip(time_values, time_counts)
            ],
        }
        json_path = sample_dir / "camera_bev_pipeline.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        if args.visualize:
            save_visualization(
                images, metas, source, outputs, cfg,
                sample_dir / "camera_bev_pipeline.png",
            )
        sample_results.append(result)
        print(
            f"[{order + 1}/{len(selected_indices)}] index={dataset_index} "
            f"images={tuple(images.shape)} total={outputs['timings_ms']['total_ms']:.3f} ms"
        )
        del images, points, matrices, metas, outputs

    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "wrapped_training_dataset_length": len(wrapped_dataset),
        "dataset_length": len(dataset),
        "num_samples": len(selected_indices),
        "uniform_dataset_indices": selected_indices,
        "warmup_iters": args.warmup_iters,
        "visualization_enabled": args.visualize,
        "timing_unit": "CUDA milliseconds",
        "timing_summary": summarize_timings(sample_results),
        "samples": sample_results,
    }
    summary_path = root / "camera_bev_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n各阶段CUDA耗时汇总（ms）：")
    for name, stats in summary["timing_summary"].items():
        print(
            f"  {name}: mean={stats['mean']:.3f}, p50={stats['p50']:.3f}, "
            f"p90={stats['p90']:.3f}"
        )
    print(f"汇总JSON：{summary_path}")


if __name__ == "__main__":
    main()
