#!/usr/bin/env python3
"""均匀抽取真实训练样本，统计spconv2 LiDAR各阶段耗时并可选可视化。

本脚本只使用 ``Spconv2Voxelization`` 和 ``SparseEncoderSpconv2``，不构建、
不执行也不绘制项目原 ``Voxelization`` 或 ``mmdet3d.ops.spconv``。

示例：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python \
      tools/visualize_spconv2_lidar_pipeline.py \
      configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_fixed_s128.yaml \
      --gpu 0 --num-samples 20 --history-sweeps 9 --visualize \
      --output-dir /tmp/spconv2_lidar_visualization
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

import spconv.pytorch as spconv
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_backbone
from mmdet3d.ops import Spconv2Voxelization
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="训练 YAML 配置")
    parser.add_argument("--num-samples", type=int, default=10, help="在数据集索引范围内均匀抽取的样本数")
    parser.add_argument("--warmup-iters", type=int, default=1, help="正式计时前使用首个样本预热的次数")
    parser.add_argument(
        "--history-sweeps",
        type=int,
        help="覆盖训练pipeline中的历史帧数；不传则沿用配置",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="可选训练权重；提供后提取encoders.lidar.backbone参数绘制真实特征",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="保存根目录")
    parser.add_argument("--visualize", action="store_true", help="保存每个样本的四联可视化图片；默认只测速")
    parser.add_argument("--max-plot-points", type=int, default=60000)
    args, overrides = parser.parse_known_args()
    if args.num_samples <= 0:
        parser.error("--num-samples 必须为正整数")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters 必须为非负整数")
    if args.history_sweeps is not None and args.history_sweeps < 0:
        parser.error("--history-sweeps 必须为非负整数")
    if args.max_plot_points <= 0:
        parser.error("--max-plot-points 必须为正整数")
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
            raise KeyError("data.train 中没有找到包含 pipeline 的内层数据集")
        current = current["dataset"]
    return current


def configure_history_sweeps(train_cfg, override):
    pipeline = inner_dataset_cfg(train_cfg)["pipeline"]
    matches = [step for step in pipeline if step.get("type") == "LoadPointsFromMultiSweeps"]
    if len(matches) != 1:
        raise RuntimeError(
            "训练pipeline应恰好包含一个LoadPointsFromMultiSweeps，实际为%d" % len(matches)
        )
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


def build_training_dataset(cfg, args):
    """只构建一次正式训练Dataset，供多个均匀样本复用。"""
    history_sweeps = configure_history_sweeps(cfg.data.train, args.history_sweeps)
    return build_dataset(cfg.data.train), history_sweeps


def uniform_dataset_indices(dataset_length, num_samples):
    """在[0, dataset_length-1]上等间隔选择索引，覆盖数据集首尾。"""
    if num_samples > dataset_length:
        raise ValueError(f"num-samples={num_samples}超过数据集长度{dataset_length}")
    return np.linspace(0, dataset_length - 1, num_samples, dtype=np.int64).tolist()


def unwrap_dataset_for_uniform_sampling(dataset):
    """去掉CBGS/Repeat等单数据集包装，避免在重采样索引上做伪均匀抽样。"""
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def load_training_points(dataset, dataset_index, history_sweeps, args, device):
    """通过正式训练Dataset pipeline获得指定索引的多帧聚合点云。"""

    original_index = dataset_index
    annotation_info = dataset.data_infos[original_index]
    raw_info = dataset.get_data_info(original_index)
    sample = dataset[dataset_index]
    if sample is None or "points" not in sample:
        raise RuntimeError("训练pipeline没有返回points")
    points = unwrap_data_container(sample["points"])
    if hasattr(points, "tensor"):
        points = points.tensor
    if not torch.is_tensor(points) or points.ndim != 2 or points.shape[1] < 5:
        raise TypeError("需要LoadPointsFromMultiSweeps输出的[N,5+]点云")

    scene_token = str(annotation_info.get("scene_token", ""))
    frame_token = str(annotation_info.get("token", raw_info.get("sample_idx", "")))
    source = {
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_outer_index": None,
        "dataset_original_index": original_index,
        "scene_token": scene_token,
        "frame_token": frame_token,
        "configured_history_sweeps": history_sweeps,
        "available_history_sweeps": len(raw_info.get("sweeps", [])),
    }
    return points.contiguous().float().to(device), source


def load_lidar_backbone_checkpoint(backbone, checkpoint_path):
    """提取LiDAR backbone权重，并兼容旧版spconv卷积核布局。"""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefixes = (
        "module.encoders.lidar.backbone.",
        "encoders.lidar.backbone.",
    )
    extracted = {}
    for name, value in state_dict.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                extracted[name[len(prefix):]] = value
                break
    if not extracted:
        raise KeyError("checkpoint中未找到encoders.lidar.backbone参数")

    # 旧版SparseBasicBlock将两个归一化层命名为bn1/bn2，当前独立
    # SparseBasicBlockSpconv2命名为norm1/norm2；层的结构和张量形状相同。
    renamed = {}
    for name, value in extracted.items():
        new_name = name.replace(".bn1.", ".norm1.").replace(".bn2.", ".norm2.")
        renamed[new_name] = value
    extracted = renamed

    #* 旧版mmdet3d.ops.spconv卷积核保存为[X,Y,Z,Cin,Cout]，当前
    #* spconv2使用[Cout,Z,Y,X,Cin]。体素化本身没有可训练参数，无需转换。
    # 已经来自spconv2的权重形状会直接保留；只有5维且目标形状匹配时才转换，
    # 避免误处理BN参数或结构不同的checkpoint。
    target_state = backbone.state_dict()
    converted_names = []
    for name, value in list(extracted.items()):
        if name not in target_state or value.shape == target_state[name].shape:
            continue
        if value.ndim == 5:
            converted = value.permute(4, 2, 1, 0, 3).contiguous()
            if converted.shape == target_state[name].shape:
                extracted[name] = converted
                converted_names.append(name)

    incompatible = backbone.load_state_dict(extracted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "LiDAR backbone权重不匹配：missing=%s, unexpected=%s"
            % (incompatible.missing_keys, incompatible.unexpected_keys)
        )
    if converted_names:
        print(
            "已将旧版spconv权重转换为spconv2布局：%d个稀疏卷积核"
            % len(converted_names)
        )


def build_spconv2_modules(cfg, device, checkpoint_path=None):
    lidar_cfg = cfg.model.encoders.lidar

    voxel_cfg = dict(lidar_cfg.voxelize)
    voxel_type = voxel_cfg.pop("type", None)
    if voxel_type != "Spconv2Voxelization":
        raise ValueError(
            "该脚本只接受type=Spconv2Voxelization，当前为%s" % voxel_type
        )
    voxelizer = Spconv2Voxelization(**voxel_cfg).to(device).train()

    backbone_cfg = dict(lidar_cfg.backbone)
    if backbone_cfg.get("type") != "SparseEncoderSpconv2":
        raise ValueError(
            "该脚本只接受type=SparseEncoderSpconv2，当前为%s"
            % backbone_cfg.get("type")
        )
    backbone = build_backbone(backbone_cfg).to(device).eval()
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint不存在：{checkpoint_path}")
        load_lidar_backbone_checkpoint(backbone, checkpoint_path)
    return voxelizer, backbone


@torch.no_grad()
def run_spconv2(points, voxelizer, backbone):
    """复现正式LiDAR分支，并用CUDA Event记录五个阶段的GPU耗时。"""

    # CUDA算子异步执行；用同一CUDA stream上的Event划分阶段，最后只同步一次。
    # 这样比每段都cuda.synchronize()更接近正式连续执行时的耗时。
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    events[0].record()

    #* ==================== 1. spconv2点云体素化 ====================
    # 将不定长点云划分到规则三维网格。voxels保存每个非空体素中的点，
    # coordinates_xyz为体素的[x,y,z]整数索引，num_points为各体素实际点数。
    # 初始体素特征：将同一体素内所有点的原始属性取平均，得到代表该体素的特征向量：[x, y, z, intensity, time_lag] 点在lidar系中的位置；反射强度，以及该店对应扫描与当前关键帧之间的时间差
    voxels, coordinates_xyz, num_points = voxelizer(points)
    events[1].record()

    #* ==================== 2. 聚合体素内点特征 ====================
    # 与BEVFusion.voxelize保持一致：对每个体素内的有效点取均值，得到
    # [非空体素数, 点特征维数]的初始体素特征；padding点不会进入分母。
    voxel_features = voxels.sum(dim=1) / num_points.type_as(voxels).view(-1, 1)
    voxel_features = voxel_features.contiguous()
    events[2].record()

    #* ==================== 3. 构造spconv2稀疏三维张量 ====================
    # 当前脚本一次只处理一个样本，因此batch索引全部为0。
    batch = torch.zeros((len(coordinates_xyz), 1), dtype=torch.int32, device=coordinates_xyz.device)
    coordinates_bxyz = torch.cat([batch, coordinates_xyz.int()], dim=1)
    # DAOcc体素化输出使用[batch,x,y,z]，spconv2要求[batch,z,y,x]。
    coordinates_bzyx = coordinates_bxyz[:, [0, 3, 2, 1]].contiguous()
    sparse = spconv.SparseConvTensor(
        voxel_features,
        coordinates_bzyx,
        backbone.sparse_shape_spconv2,
        1,
    )
    events[3].record()

    #* ==================== 4. spconv2稀疏卷积backbone ====================
    # conv_input和encoder_layers在非空体素位置执行三维稀疏卷积；
    # conv_out进一步编码并下采样，输出仍是SparseConvTensor，而不是BEV。
    x = backbone.conv_input(sparse)
    for layer in backbone.encoder_layers:
        x = layer(x)
    output = backbone.conv_out(x)
    events[4].record()

    #* ==================== 5. 稠密化并生成LiDAR BEV特征 ====================
    # dense()把稀疏体素填回完整网格，得到[N,C,D,H,W]；空体素位置补0。
    # output：spconv2的SparseConvensor：内部主要保存：output.features   # 非空位置的特征，形状约为 [M, C]
    # output.indices    # 每条特征所在位置，形状为 [M, 4]，顺序为 [batch, z, y, x]
    # 初始 shape 会保存在 SparseConvTensor.spatial_shape 中。
    spatial_features = output.dense() # 将稀疏体素填回到完整网格中
    n, c, d, h, w = spatial_features.shape
    # 与SparseEncoderSpconv2.forward完全一致：把高度D折叠到通道维，
    # 得到供后续BEV融合使用的[N,C*D,H,W]二维LiDAR BEV特征。
    bev_features = spatial_features.reshape(n, c * d, h, w) # 高度维度直接折叠到通道维度
    events[5].record()
    events[-1].synchronize()

    stage_names = (
        "voxelization_ms",
        "voxel_feature_reduction_ms",
        "sparse_tensor_construction_ms",
        "sparse_backbone_ms",
        "dense_and_bev_ms",
    )
    timings = {
        name: float(events[index].elapsed_time(events[index + 1]))
        for index, name in enumerate(stage_names)
    }
    timings["total_ms"] = float(events[0].elapsed_time(events[-1]))

    # 将spconv2的[batch,z,y,x]索引转回便于理解的[batch,x,y,z]。
    # 此操作只服务于坐标图，放在计时区间外，不计入正式LiDAR分支耗时。
    output_bxyz = output.indices[:, [0, 3, 2, 1]].contiguous()

    return {
        "voxel_coordinates_xyz": coordinates_xyz,
        "voxel_num_points": num_points,
        "backbone_coordinates_xyz": output_bxyz[:, 1:],
        "dense_3d_features": spatial_features,
        "bev_features": bev_features,
        "timings_ms": timings,
    }


def physical_centers(coordinates_xyz, point_range, voxel_size, stride=1):
    coordinates = np.asarray(coordinates_xyz, dtype=np.float64)
    if coordinates.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    mins = np.asarray(point_range[:3], dtype=np.float64)
    sizes = np.asarray(voxel_size, dtype=np.float64) * stride
    return mins + (coordinates + 0.5) * sizes


def uniform_subsample(array, limit, seed):
    """从全部点统一抽样，使当前帧/历史帧绘制比例接近真实点数比例。"""
    if len(array) <= limit:
        return array
    rng = np.random.default_rng(seed)
    return array[rng.choice(len(array), limit, replace=False)]


def save_visualization(points, outputs, cfg, path, limit, seed):
    points_np = points.detach().cpu().numpy()
    plotted = uniform_subsample(points_np, limit, seed)
    current = np.isclose(plotted[:, 4], 0.0, atol=1e-8)
    current_total = int(np.isclose(points_np[:, 4], 0.0, atol=1e-8).sum())
    history_total = int(len(points_np) - current_total)

    point_range = list(cfg.model.encoders.lidar.voxelize.point_cloud_range)
    voxel_size = list(cfg.model.encoders.lidar.voxelize.voxel_size)
    voxel_xyz = outputs["voxel_coordinates_xyz"].detach().cpu().numpy()
    backbone_xyz = outputs["backbone_coordinates_xyz"].detach().cpu().numpy()
    voxel_xy = physical_centers(voxel_xyz, point_range, voxel_size)[:, :2]
    backbone_xy = physical_centers(backbone_xyz, point_range, voxel_size, stride=8)[:, :2]
    voxel_xy = uniform_subsample(voxel_xy, limit, seed + 1)
    backbone_xy = uniform_subsample(backbone_xy, limit, seed + 2)
    
    #* ---------------------------------------------------#
    #* 
    # 对通道取绝对值均值，得到每个BEV网格的整体激活强度。
    # 这里只用于可视化，不参与模型推理，也不改变正式网络输出。
    bev = outputs["bev_features"][0].detach().float().cpu()
    bev_heatmap = bev.abs().mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(28, 7), constrained_layout=True)

    # 先绘制历史帧，再将半透明的当前帧绘制在上层，使当前帧轮廓更容易辨认。
    # 两类点仍从同一次统一抽样得到，因此显示数量保持真实点数比例。
    axes[0].scatter(
        plotted[~current, 0], plotted[~current, 1], s=0.18,
        c="tab:orange", alpha=0.55,
        label=f"history ({history_total:,} points)", zorder=2,
    )
    axes[0].scatter(
        plotted[current, 0], plotted[current, 1], s=0.32,
        c="tab:blue", alpha=0.55,
        label=f"current ({current_total:,} points)", zorder=3,
    )
    axes[0].set_title("Training input: current + history LiDAR")
    axes[0].legend(markerscale=8)

    axes[1].scatter(voxel_xy[:, 0], voxel_xy[:, 1], s=0.25, c="tab:green")
    axes[1].set_title(f"spconv2 initial voxels ({len(voxel_xyz):,})")

    axes[2].scatter(backbone_xy[:, 0], backbone_xy[:, 1], s=0.8, c="tab:purple")
    axes[2].set_title(f"spconv2 backbone output coordinates ({len(backbone_xyz):,})")

    heatmap = axes[3].imshow(
        bev_heatmap,
        origin="lower",
        extent=[point_range[0], point_range[3], point_range[1], point_range[4]],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    axes[3].set_title(
        f"LiDAR BEV |feature| mean ({bev.shape[0]} channels)"
    )
    fig.colorbar(heatmap, ax=axes[3], fraction=0.046, pad=0.04, label="mean |activation|")

    for axis in axes:
        axis.set_xlim(point_range[0], point_range[3])
        axis.set_ylim(point_range[1], point_range[4])
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize_timings(sample_results):
    """汇总各阶段耗时的均值、P50、P90、最小值和最大值。"""
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
        raise RuntimeError("spconv2可视化需要CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config, overrides)
    wrapped_dataset, history_sweeps = build_training_dataset(cfg, args)
    dataset = unwrap_dataset_for_uniform_sampling(wrapped_dataset)
    selected_indices = uniform_dataset_indices(len(dataset), args.num_samples)
    voxelizer, backbone = build_spconv2_modules(cfg, device, args.checkpoint)
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 使用首个均匀样本预热spconv2/JIT算法选择，预热结果不计入统计。
    if args.warmup_iters:
        warmup_points, _ = load_training_points(
            dataset, selected_indices[0], history_sweeps, args, device)
        for _ in range(args.warmup_iters):
            warmup_outputs = run_spconv2(warmup_points, voxelizer, backbone)
            del warmup_outputs
        del warmup_points
        torch.cuda.empty_cache()

    sample_results = []
    for order, dataset_index in enumerate(selected_indices):
        points, source = load_training_points(
            dataset, dataset_index, history_sweeps, args, device)
        outputs = run_spconv2(points, voxelizer, backbone)

        scene_token = source["scene_token"] or "scene_unknown"
        frame_token = source["frame_token"] or f"frame_index_{source['dataset_original_index']}"
        sample_dir = root / f"{scene_token}_{frame_token}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        time_values, time_counts = torch.unique(
            points[:, 4].cpu(), sorted=True, return_counts=True
        )
        result = {
            "sample_order": order,
            "source": {**source, "output_directory": str(sample_dir)},
            "lidar_backbone_checkpoint": (
                str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None
            ),
            "num_input_points": len(points),
            "num_spconv2_voxels": len(outputs["voxel_coordinates_xyz"]),
            "num_spconv2_backbone_output_coordinates": len(
                outputs["backbone_coordinates_xyz"]
            ),
            "dense_3d_feature_shape": list(outputs["dense_3d_features"].shape),
            "bev_feature_shape": list(outputs["bev_features"].shape),
            "timings_ms": outputs["timings_ms"],
            "time_lag_groups": [
                {"seconds": float(value), "num_points": int(count)}
                for value, count in zip(time_values, time_counts)
            ],
            "coordinate_convention": {
                "Spconv2Voxelization_output": "[x, y, z]",
                "SparseEncoderSpconv2_internal": "[batch, z, y, x]",
                "visualized_backbone_output": "[x, y, z] after adapter",
            },
        }
        json_path = sample_dir / "spconv2_lidar_pipeline.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        if args.visualize:
            image_path = sample_dir / "spconv2_lidar_pipeline.png"
            save_visualization(
                points, outputs, cfg, image_path, args.max_plot_points,
                args.seed + order,
            )
        sample_results.append(result)
        print(
            f"[{order + 1}/{len(selected_indices)}] index={dataset_index} "
            f"points={len(points)} total={outputs['timings_ms']['total_ms']:.3f} ms"
        )
        del points, outputs

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
    summary_path = root / "spconv2_lidar_pipeline_summary.json"
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
