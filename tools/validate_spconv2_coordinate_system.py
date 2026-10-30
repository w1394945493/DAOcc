#!/usr/bin/env python3
"""用 DAOcc 的真实训练点云验证原实现与 spconv2 的物理坐标系是否一致。

示例：
  python tools/validate_spconv2_coordinate_system.py \
      configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel.yaml \
      --sample-index 0 --history-sweeps 9 --output-dir /tmp/coord_check

脚本直接构建 ``cfg.data.train`` 并读取一个训练样本，因此点云会经过与
``tools/train.py`` 相同的流程：当前关键帧加载、历史 sweep 选择、历史点云变换到
当前 LiDAR 坐标系、拼接、三维增强、范围过滤和 PointShuffle。不会再生成随机点云，
也不再直接读取一份绕过 Dataset pipeline 的单帧 .bin/.npy 文件。
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from torchpack.utils.config import Config as TorchpackConfig

from mmdet3d.datasets import build_dataset
from mmdet3d.models.backbones.sparse_encoder import SparseEncoder
from mmdet3d.models.backbones.sparse_encoder_spconv2 import SparseEncoderSpconv2
from mmdet3d.ops import Spconv2Voxelization, Voxelization
from mmdet3d.ops import spconv as old_spconv
from mmdet3d.utils import recursive_eval
import spconv.pytorch as spconv2


VOXEL_SIZE = [0.075, 0.075, 0.2]
POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
SPARSE_SHAPE = [1440, 1440, 41]
MAX_NUM_POINTS = 10
MAX_VOXELS = 120000
XY_BACKBONE_STRIDE = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="训练 YAML 配置；按 tools/train.py 的方式递归加载")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="训练数据集的外层索引；启用 CBGS 时会先映射到原始场景索引",
    )
    parser.add_argument(
        "--history-sweeps",
        type=int,
        help="覆盖 LoadPointsFromMultiSweeps.sweeps_num；不传则沿用训练配置",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="保存根目录；结果写入其下的 场景token_帧token 子目录",
    )
    parser.add_argument(
        "--max-plot-points", type=int, default=60000, help="每个子图最多绘制的点数"
    )
    args, overrides = parser.parse_known_args()
    if args.sample_index < 0:
        parser.error("--sample-index 必须为非负整数")
    if args.history_sweeps is not None and args.history_sweeps < 0:
        parser.error("--history-sweeps 必须为非负整数")
    return args, overrides


def load_config(path, overrides):
    """与 tools/train.py 一样递归合并 YAML，并应用命令行配置覆盖项。"""
    raw = TorchpackConfig()
    raw.load(path, recursive=True)
    raw.update(overrides)
    return Config(recursive_eval(raw), filename=path)


def inner_dataset_cfg(train_cfg):
    """穿过 CBGSDataset 等包装器，找到真正包含 pipeline 的数据集配置。"""
    current = train_cfg
    while "pipeline" not in current:
        if "dataset" not in current:
            raise KeyError("data.train 中没有找到包含 pipeline 的内层数据集")
        current = current["dataset"]
    return current


def configure_history_sweeps(train_cfg, override):
    """读取或覆盖正式 pipeline 的历史 sweep 数量。"""
    pipeline = inner_dataset_cfg(train_cfg)["pipeline"]
    matches = [step for step in pipeline if step.get("type") == "LoadPointsFromMultiSweeps"]
    if len(matches) != 1:
        raise RuntimeError(
            "训练 pipeline 应恰好包含一个 LoadPointsFromMultiSweeps，实际为 %d" % len(matches)
        )
    if override is not None:
        matches[0]["sweeps_num"] = override
    return int(matches[0]["sweeps_num"])


def unwrap_data_container(value):
    """取出 Collect3D/DefaultFormatBundle3D 包装后的实际数据。"""
    while value.__class__.__name__ == "DataContainer":
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        while value.__class__.__name__ == "DataContainer":
            value = value.data
    return value


def load_training_points(cfg, args, device):
    """执行真实训练 Dataset pipeline，返回当前帧与历史帧聚合后的点云。"""
    history_sweeps = configure_history_sweeps(cfg.data.train, args.history_sweeps)
    dataset = build_dataset(cfg.data.train)
    if args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index={args.sample_index} 超出训练数据集长度 {len(dataset)}"
        )

    # CBGSDataset 的外层索引会先映射到原始场景；记录映射便于复现实验。
    original_index = args.sample_index
    base_dataset = dataset
    if hasattr(dataset, "sample_indices") and getattr(dataset, "resample", False):
        original_index = int(dataset.sample_indices[args.sample_index])
        base_dataset = dataset.dataset
    elif hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset

    raw_info = base_dataset.get_data_info(original_index)
    annotation_info = base_dataset.data_infos[original_index]
    scene_token = str(annotation_info.get("scene_token", ""))
    frame_token = str(
        annotation_info.get("token", raw_info.get("sample_idx", ""))
    )
    sample = dataset[args.sample_index]
    if sample is None or "points" not in sample:
        raise RuntimeError("训练 pipeline 没有返回 points；样本可能被过滤")
    points = unwrap_data_container(sample["points"])
    if hasattr(points, "tensor"):
        points = points.tensor
    if not torch.is_tensor(points) or points.ndim != 2 or points.shape[1] < 3:
        raise TypeError(f"训练 pipeline 返回的 points 应为 [N,3+] Tensor，实际为 {type(points)}")

    # 第5维是 LoadPointsFromMultiSweeps 写入的时间差：当前帧为0，历史帧为正值。
    time_lags = []
    if points.shape[1] >= 5:
        values, counts = torch.unique(points[:, 4].cpu(), sorted=True, return_counts=True)
        time_lags = [
            {"seconds": float(value), "num_points": int(count)}
            for value, count in zip(values, counts)
        ]
    source = {
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_outer_index": args.sample_index,
        "dataset_original_index": original_index,
        "sample_idx": str(raw_info.get("sample_idx", "")),
        "scene_token": scene_token,
        "frame_token": frame_token,
        "keyframe_lidar": str(
            raw_info.get("pts_filename", raw_info.get("lidar_path", ""))
        ),
        "available_history_sweeps": len(raw_info.get("sweeps", [])),
        "configured_history_sweeps": history_sweeps,
        "time_lag_groups": time_lags,
    }
    return points.contiguous().float().to(device), source


def make_backbone(cls):
    return cls(
        in_channels=5,
        sparse_shape=SPARSE_SHAPE,
        output_channels=128,
        encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
        encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
        block_type="basicblock",
    )


def tensor_coordinate_set(coordinates):
    return set(map(tuple, coordinates.detach().cpu().numpy().tolist()))


def xyz_centers(coordinates_xyz, voxel_size=VOXEL_SIZE, stride=1):
    coordinates_xyz = np.asarray(coordinates_xyz, dtype=np.float64)
    if coordinates_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    mins = np.asarray(POINT_CLOUD_RANGE[:3], dtype=np.float64)
    sizes = np.asarray(voxel_size, dtype=np.float64) * stride
    return mins + (coordinates_xyz + 0.5) * sizes


def compare_voxelizers(points):
    kwargs = dict(
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE,
        max_num_points=MAX_NUM_POINTS,
        max_voxels=[MAX_VOXELS, MAX_VOXELS],
    )
    old = Voxelization(**kwargs).to(points.device).eval()
    new = Spconv2Voxelization(**kwargs).to(points.device).eval()
    with torch.no_grad():
        old_voxels, old_coords, old_sizes = old(points)
        new_voxels, new_coords, new_sizes = new(points)

    old_set = tensor_coordinate_set(old_coords)
    new_set = tensor_coordinate_set(new_coords)
    common = old_set & new_set
    old_only = old_set - new_set
    new_only = new_set - old_set

    # 相同整数 XYZ 索引通过同一 voxel_size/range 转为物理中心，误差应为 0。
    common_array = np.asarray(sorted(common), dtype=np.int32).reshape(-1, 3)
    old_centers = xyz_centers(common_array)
    new_centers = xyz_centers(common_array)
    center_error = (
        float(np.abs(old_centers - new_centers).max()) if len(common_array) else None
    )
    return {
        "old_voxels": old_voxels,
        "new_voxels": new_voxels,
        "old_sizes": old_sizes,
        "new_sizes": new_sizes,
        "old_set": old_set,
        "new_set": new_set,
        "common": common,
        "old_only": old_only,
        "new_only": new_only,
        "center_error": center_error,
    }


def compare_backbones(common_voxel_coordinates, device):
    if not common_voxel_coordinates:
        raise RuntimeError("新旧体素器没有公共坐标，无法验证 backbone")

    coordinates_xyz = torch.tensor(
        sorted(common_voxel_coordinates), dtype=torch.int32, device=device
    )
    batch = torch.zeros((len(coordinates_xyz), 1), dtype=torch.int32, device=device)
    coordinates_bxyz = torch.cat([batch, coordinates_xyz], dim=1)

    # 使用同一批坐标和同一组特征，单独验证 backbone 对坐标轴的解释。
    centers = torch.tensor(
        xyz_centers(coordinates_xyz.cpu().numpy()), dtype=torch.float32, device=device
    )
    features = torch.cat(
        [centers / 54.0, torch.ones((len(centers), 2), device=device)], dim=1
    )

    old = make_backbone(SparseEncoder).to(device).eval()
    new = make_backbone(SparseEncoderSpconv2).to(device).eval()
    with torch.no_grad():
        old_tensor = old_spconv.SparseConvTensor(
            features, coordinates_bxyz, old.sparse_shape, 1
        )
        x_old = old.conv_input(old_tensor)
        for layer in old.encoder_layers:
            x_old = layer(x_old)
        out_old = old.conv_out(x_old)

        coordinates_bzyx = coordinates_bxyz[:, [0, 3, 2, 1]].contiguous()
        new_tensor = spconv2.SparseConvTensor(
            features, coordinates_bzyx, new.sparse_shape_spconv2, 1
        )
        x_new = new.conv_input(new_tensor)
        for layer in new.encoder_layers:
            x_new = layer(x_new)
        out_new = new.conv_out(x_new)

    old_bxyz = out_old.indices
    new_bxyz = out_new.indices[:, [0, 3, 2, 1]].contiguous()
    old_set = tensor_coordinate_set(old_bxyz[:, 1:])
    new_set = tensor_coordinate_set(new_bxyz[:, 1:])
    return {
        "old_set": old_set,
        "new_set": new_set,
        "common": old_set & new_set,
        "old_only": old_set - new_set,
        "new_only": new_set - old_set,
    }


def sample_xy(coordinate_set, stride, limit, seed):
    if not coordinate_set:
        return np.empty((0, 2))
    array = np.asarray(list(coordinate_set), dtype=np.int32)
    if len(array) > limit:
        rng = np.random.default_rng(seed)
        array = array[rng.choice(len(array), limit, replace=False)]
    return xyz_centers(array, stride=stride)[:, :2]


def save_visualization(points, voxel, backbone, output_path, limit, seed):
    points_cpu = points.detach().cpu().numpy()

    # LoadPointsFromMultiSweeps 将第5维设置为相对当前关键帧的时间差：
    # 当前帧为0，历史帧大于0。分别抽样后使用不同颜色，避免数量更多的历史点
    # 完全遮住当前帧；两类合计仍不超过 max_plot_points。
    if points_cpu.shape[1] >= 5:
        current_mask = np.isclose(points_cpu[:, 4], 0.0, atol=1e-8)
    else:
        current_mask = np.ones(len(points_cpu), dtype=bool)
    current_xy_all = points_cpu[current_mask, :2]
    history_xy_all = points_cpu[~current_mask, :2]

    def subsample_xy(array, max_count, sample_seed):
        if len(array) <= max_count:
            return array
        rng = np.random.default_rng(sample_seed)
        return array[rng.choice(len(array), max_count, replace=False)]

    current_budget = min(len(current_xy_all), max(1, limit // 2))
    history_budget = min(len(history_xy_all), max(0, limit - current_budget))
    # 没有历史点时，将剩余绘图预算全部交给当前帧。
    if history_budget == 0:
        current_budget = min(len(current_xy_all), limit)
    current_xy = subsample_xy(current_xy_all, current_budget, seed)
    history_xy = subsample_xy(history_xy_all, history_budget, seed + 1)

    # 原来的左下角只在坐标集合不一致时才有内容；当前验证结果为空，因此删除该图，
    # 改为一行三图：时序点云、新旧体素中心、新旧backbone输出坐标。
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    if len(history_xy):
        axes[0].scatter(
            history_xy[:, 0], history_xy[:, 1], s=0.18, c="tab:orange",
            alpha=0.55, label=f"history sweeps ({len(history_xy_all):,} points)"
        )
    if len(current_xy):
        axes[0].scatter(
            current_xy[:, 0], current_xy[:, 1], s=0.28, c="tab:blue",
            alpha=0.9, label=f"current keyframe ({len(current_xy_all):,} points)"
        )
    axes[0].set_title("Aggregated LiDAR: current vs. history")
    axes[0].legend(markerscale=8)

    old_xy = sample_xy(voxel["old_set"], 1, limit, seed)
    new_xy = sample_xy(voxel["new_set"], 1, limit, seed)
    axes[1].scatter(old_xy[:, 0], old_xy[:, 1], s=0.4, c="tab:blue", label="old")
    axes[1].scatter(
        new_xy[:, 0], new_xy[:, 1], s=0.15, c="tab:orange", label="spconv2", alpha=0.7
    )
    axes[1].set_title("Voxel centers in the shared physical XY frame")
    axes[1].legend(markerscale=8)

    old_bev = sample_xy(backbone["old_set"], XY_BACKBONE_STRIDE, limit, seed)
    new_bev = sample_xy(backbone["new_set"], XY_BACKBONE_STRIDE, limit, seed)
    axes[2].scatter(old_bev[:, 0], old_bev[:, 1], s=1.0, c="tab:blue", label="old")
    axes[2].scatter(
        new_bev[:, 0], new_bev[:, 1], s=0.35, c="tab:orange", label="spconv2", alpha=0.7
    )
    axes[2].set_title("Backbone output coordinates in physical BEV XY")
    axes[2].legend(markerscale=5)

    for axis in axes:
        axis.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
        axis.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args, overrides = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("旧版稀疏卷积与 spconv2 对比需要 CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, overrides)
    points, source = load_training_points(cfg, args, device)

    # 每个真实样本单独保存，目录名同时包含场景token与当前帧token：
    #   保存根目录/<scene_token>_<frame_token>/coordinate_validation.json
    #   保存根目录/<scene_token>_<frame_token>/coordinate_system_comparison.png
    # 标注缺少token时使用可追踪的索引占位，避免多个样本覆盖同一目录。
    scene_token = source["scene_token"] or "scene_unknown"
    frame_token = source["frame_token"] or f"frame_index_{source['dataset_original_index']}"
    sample_output_dir = output_root / f"{scene_token}_{frame_token}"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    source["output_directory"] = str(sample_output_dir)

    voxel = compare_voxelizers(points)
    backbone = compare_backbones(voxel["common"], device)

    result = {
        "source": source,
        "num_input_points": len(points),
        "point_feature_dim": int(points.shape[1]),
        "coordinate_convention": {
            "DAOcc_voxelizer_output": "[x, y, z]",
            "spconv2_raw_output": "[z, y, x]",
            "Spconv2Voxelization_output_after_adapter": "[x, y, z]",
            "physical_range_xyzxyz": POINT_CLOUD_RANGE,
            "voxel_size_xyz": VOXEL_SIZE,
            "backbone_xy_stride": XY_BACKBONE_STRIDE,
        },
        "voxel_coordinates": {
            "old_count": len(voxel["old_set"]),
            "spconv2_count": len(voxel["new_set"]),
            "common_count": len(voxel["common"]),
            "old_only_count": len(voxel["old_only"]),
            "spconv2_only_count": len(voxel["new_only"]),
            "sets_equal": voxel["old_set"] == voxel["new_set"],
            "common_physical_center_max_error_meter": voxel["center_error"],
        },
        "backbone_output_coordinates": {
            "old_count": len(backbone["old_set"]),
            "spconv2_count": len(backbone["new_set"]),
            "common_count": len(backbone["common"]),
            "old_only_count": len(backbone["old_only"]),
            "spconv2_only_count": len(backbone["new_only"]),
            "sets_equal_after_axis_adapter": backbone["old_set"] == backbone["new_set"],
        },
    }
    json_path = sample_output_dir / "coordinate_validation.json"
    image_path = sample_output_dir / "coordinate_system_comparison.png"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # *-------------------------------------------------------------------------------#
    # * 结果可视化
    save_visualization(points, voxel, backbone, image_path, args.max_plot_points, args.seed)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nJSON: {json_path}")
    print(f"可视化: {image_path}")


if __name__ == "__main__":
    main()
