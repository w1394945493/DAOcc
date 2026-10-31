#!/usr/bin/env python3
"""逐项比较 DAOcc 原 LiDAR 分支与 spconv2 替换实现。

本脚本使用正式训练 Dataset pipeline 取得同一帧多 sweep 聚合点云，并比较：
1. 新旧体素坐标集合；
2. 公共体素的点数和均值特征；
3. LiDAR backbone 各阶段的稀疏坐标及特征；
4. 最终 BEV 特征及其空间热力图；
5. 同一标量目标反向传播后的输入特征和参数梯度。

为使数值比较有意义，先构建旧 SparseEncoder，再把它的全部参数和 BN 状态转换并
加载到 SparseEncoderSpconv2。两条分支不会使用两套互不相关的随机权重。

示例：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python \
    tools/validate_spconv2_lidar_equivalence.py \
    configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel.yaml \
    --gpu 0 --sample-index 0 --history-sweeps 9 \
    --output-dir /tmp/spconv2_lidar_equivalence
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

import spconv.pytorch as spconv2
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_backbone
from mmdet3d.ops import Spconv2Voxelization, Voxelization
from mmdet3d.ops import spconv as old_spconv
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="spconv2 训练配置")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--history-sweeps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-backward", action="store_true", help="只比较前向，跳过梯度比较"
    )
    args, overrides = parser.parse_known_args()
    if args.sample_index < 0:
        parser.error("--sample-index 必须为非负整数")
    if args.history_sweeps is not None and args.history_sweeps < 0:
        parser.error("--history-sweeps 必须为非负整数")
    return args, overrides


def load_config(path, overrides):
    raw = TorchpackConfig()
    raw.load(path, recursive=True)
    raw.update(overrides)
    return Config(recursive_eval(raw), filename=path)


def inner_dataset_cfg(train_cfg):
    current = train_cfg
    while "pipeline" not in current:
        current = current["dataset"]
    return current


def configure_history_sweeps(train_cfg, override):
    pipeline = inner_dataset_cfg(train_cfg)["pipeline"]
    matches = [x for x in pipeline if x.get("type") == "LoadPointsFromMultiSweeps"]
    if len(matches) != 1:
        raise RuntimeError("训练 pipeline 应恰好包含一个 LoadPointsFromMultiSweeps")
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


def load_training_points(cfg, args, device):
    """按正式训练 pipeline 加载当前帧和历史帧聚合后的点云。"""
    sweeps = configure_history_sweeps(cfg.data.train, args.history_sweeps)
    wrapped = build_dataset(cfg.data.train)
    dataset = wrapped
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index 超出基础训练数据集长度 {len(dataset)}")

    info = dataset.data_infos[args.sample_index]
    raw_info = dataset.get_data_info(args.sample_index)
    sample = dataset[args.sample_index]
    if sample is None or "points" not in sample:
        raise RuntimeError("训练 pipeline 没有返回 points")
    points = unwrap_data_container(sample["points"])
    if hasattr(points, "tensor"):
        points = points.tensor
    if not torch.is_tensor(points) or points.ndim != 2:
        raise TypeError("points 应为二维 Tensor")

    source = {
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_index": args.sample_index,
        "scene_token": str(info.get("scene_token", "")),
        "frame_token": str(info.get("token", raw_info.get("sample_idx", ""))),
        "history_sweeps": sweeps,
        "num_input_points": int(len(points)),
        "point_feature_dim": int(points.shape[1]),
    }
    return points.contiguous().float().to(device), source


def build_modules(cfg, device):
    """从同一配置派生新旧体素器和新旧 backbone。"""
    lidar_cfg = cfg.model.encoders.lidar

    voxel_cfg = dict(lidar_cfg.voxelize)
    voxel_cfg.pop("type", None)
    # 当前目标是排查训练精度下降，因此使用 max_voxels[0]（训练上限），
    # 而不是 eval() 对应的 max_voxels[1]。backbone 仍使用 eval()，以冻结
    # BN 状态并单独观察两种稀疏卷积实现的确定性数值差异。
    old_voxelizer = Voxelization(**voxel_cfg).to(device).train()
    new_voxelizer = Spconv2Voxelization(**voxel_cfg).to(device).train()

    backbone_cfg = dict(lidar_cfg.backbone)
    backbone_cfg["type"] = "SparseEncoder"
    old_backbone = build_backbone(backbone_cfg).to(device).eval()
    backbone_cfg["type"] = "SparseEncoderSpconv2"
    new_backbone = build_backbone(backbone_cfg).to(device).eval()
    copy_old_state_to_spconv2(old_backbone, new_backbone)
    return old_voxelizer, new_voxelizer, old_backbone, new_backbone


def copy_old_state_to_spconv2(old_model, new_model):
    """把旧版权重、BN 参数及 running stats 精确转换到 spconv2。"""
    old_state = old_model.state_dict()
    converted = {}
    for new_name, target in new_model.state_dict().items():
        old_name = new_name.replace(".norm1.", ".bn1.").replace(".norm2.", ".bn2.")
        if old_name not in old_state:
            raise KeyError(f"旧 backbone 中找不到与 {new_name} 对应的 {old_name}")
        value = old_state[old_name]
        if value.shape != target.shape and value.ndim == 5:
            # 旧版 [X,Y,Z,Cin,Cout] -> spconv2 [Cout,Z,Y,X,Cin]。
            value = value.permute(4, 2, 1, 0, 3).contiguous()
        if value.shape != target.shape:
            raise RuntimeError(
                f"状态形状不匹配：{old_name} {tuple(value.shape)} -> "
                f"{new_name} {tuple(target.shape)}"
            )
        converted[new_name] = value
    new_model.load_state_dict(converted, strict=True)


def coordinate_rows(coords):
    return [tuple(int(v) for v in row) for row in coords.detach().cpu().tolist()]


def coordinate_overlap(old_coords, new_coords):
    old_set, new_set = set(coordinate_rows(old_coords)), set(coordinate_rows(new_coords))
    common = old_set & new_set
    union = old_set | new_set
    return {
        "old_count": len(old_set),
        "spconv2_count": len(new_set),
        "common_count": len(common),
        "old_only_count": len(old_set - new_set),
        "spconv2_only_count": len(new_set - old_set),
        "jaccard": float(len(common) / len(union)) if union else 1.0,
        "old_coverage": float(len(common) / len(old_set)) if old_set else 1.0,
        "spconv2_coverage": float(len(common) / len(new_set)) if new_set else 1.0,
        "sets_equal": old_set == new_set,
    }


def tensor_stats(tensor):
    if tensor is None:
        return {"available": False, "reason": "该实现未保留此张量的梯度"}
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    valid = x[finite]
    return {
        "available": True,
        "shape": list(x.shape),
        "dtype": str(tensor.dtype),
        "finite_fraction": float(finite.float().mean()),
        "mean": float(valid.mean()) if valid.numel() else None,
        "std": float(valid.std(unbiased=False)) if valid.numel() else None,
        "abs_max": float(valid.abs().max()) if valid.numel() else None,
        "l2_norm": float(torch.linalg.vector_norm(valid)) if valid.numel() else None,
    }


def align_common_features(old_coords, old_features, new_coords, new_features):
    """按整数坐标对齐公共位置，不能依赖两套算子的内部输出顺序。"""
    if old_features is None or new_features is None:
        return None, None, []
    old_rows, new_rows = coordinate_rows(old_coords), coordinate_rows(new_coords)
    old_map = {coord: i for i, coord in enumerate(old_rows)}
    new_map = {coord: i for i, coord in enumerate(new_rows)}
    common = sorted(set(old_map) & set(new_map))
    if not common:
        return None, None, []
    old_idx = torch.tensor([old_map[x] for x in common], device=old_features.device)
    new_idx = torch.tensor([new_map[x] for x in common], device=new_features.device)
    return old_features[old_idx], new_features[new_idx], common


def feature_difference(old_values, new_values):
    if old_values is None:
        return {"comparable": False, "reason": "没有公共坐标"}
    old_values = old_values.detach().float()
    new_values = new_values.detach().float()
    if old_values.shape != new_values.shape:
        return {
            "comparable": False,
            "reason": "公共坐标上的通道形状不同",
            "old_shape": list(old_values.shape),
            "spconv2_shape": list(new_values.shape),
        }
    delta = old_values - new_values
    old_flat, new_flat = old_values.reshape(-1), new_values.reshape(-1)
    denom = torch.linalg.vector_norm(old_flat) * torch.linalg.vector_norm(new_flat)
    cosine = float(torch.dot(old_flat, new_flat) / denom) if float(denom) > 0 else None
    return {
        "comparable": True,
        "num_values": int(delta.numel()),
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "abs_max": float(delta.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(old_flat).clamp_min(1e-12)
        ),
        "cosine_similarity": cosine,
    }


def voxelize_and_reduce(points, voxelizer):
    voxels, coords_xyz, counts = voxelizer(points)
    features = voxels.sum(dim=1) / counts.type_as(voxels).view(-1, 1)
    return features.contiguous(), coords_xyz.int().contiguous(), counts


def run_old_backbone(features, coords_xyz, backbone, retain_input_gradient=False):
    """逐阶段执行旧 SparseEncoder，并保留稀疏输出用于对齐。"""
    batch = torch.zeros((len(coords_xyz), 1), dtype=torch.int32, device=coords_xyz.device)
    coords_bxyz = torch.cat([batch, coords_xyz], dim=1)
    x = old_spconv.SparseConvTensor(features, coords_bxyz, backbone.sparse_shape, 1)
    sparse_input_features = x.features
    if retain_input_gradient and sparse_input_features.requires_grad:
        # 部分稀疏张量实现会包装传入 features；直接保留内部实际参与卷积的张量梯度。
        sparse_input_features.retain_grad()
    stages = {}
    x = backbone.conv_input(x)
    stages["conv_input"] = x
    for index, layer in enumerate(backbone.encoder_layers, 1):
        x = layer(x)
        stages[f"encoder_layer{index}"] = x
    x = backbone.conv_out(x)
    stages["conv_out"] = x
    dense = x.dense()  # 旧版返回 [N,C,H,W,D]。
    n, c, h, w, d = dense.shape
    bev = dense.permute(0, 1, 4, 2, 3).contiguous().reshape(n, c * d, h, w)
    if retain_input_gradient:
        return stages, bev, sparse_input_features
    return stages, bev


def run_new_backbone(features, coords_xyz, backbone, retain_input_gradient=False):
    """逐阶段执行 spconv2，并把坐标转换回 [batch,x,y,z] 后比较。"""
    batch = torch.zeros((len(coords_xyz), 1), dtype=torch.int32, device=coords_xyz.device)
    coords_bxyz = torch.cat([batch, coords_xyz], dim=1)
    coords_bzyx = coords_bxyz[:, [0, 3, 2, 1]].contiguous()
    x = spconv2.SparseConvTensor(features, coords_bzyx, backbone.sparse_shape_spconv2, 1)
    sparse_input_features = x.features
    if retain_input_gradient and sparse_input_features.requires_grad:
        # spconv2 内部特征可能不是调用方传入的 leaf，必须在前向前显式 retain_grad()。
        sparse_input_features.retain_grad()
    stages = {}
    x = backbone.conv_input(x)
    stages["conv_input"] = x
    for index, layer in enumerate(backbone.encoder_layers, 1):
        x = layer(x)
        stages[f"encoder_layer{index}"] = x
    x = backbone.conv_out(x)
    stages["conv_out"] = x
    dense = x.dense()  # spconv2 返回 [N,C,D,H,W]。
    n, c, d, h, w = dense.shape
    
    bev = dense.reshape(n, c * d, h, w)
    bev = bev.permute(0,1,3,2).contiguous() # 返回 [N,C*D,W,H]
    
    if retain_input_gradient:
        return stages, bev, sparse_input_features
    return stages, bev


def sparse_parts(tensor, implementation):
    if implementation == "old":
        return tensor.indices.int(), tensor.features
    # spconv2 [batch,z,y,x] -> DAOcc/旧实现 [batch,x,y,z]。
    return tensor.indices[:, [0, 3, 2, 1]].int().contiguous(), tensor.features


def compare_voxels(old_result, new_result):
    old_features, old_coords, old_counts = old_result
    new_features, new_coords, new_counts = new_result
    old_f, new_f, common = align_common_features(
        old_coords, old_features, new_coords, new_features
    )
    old_n, new_n, _ = align_common_features(
        old_coords, old_counts[:, None].float(), new_coords, new_counts[:, None].float()
    )
    count_delta = None
    if old_n is not None:
        diff = old_n - new_n
        count_delta = {
            "equal_fraction": float((diff == 0).float().mean()),
            "mean_absolute_difference": float(diff.abs().mean()),
            "max_absolute_difference": float(diff.abs().max()),
        }
    return {
        "coordinates": coordinate_overlap(old_coords, new_coords),
        "common_voxel_count": len(common),
        "point_counts_on_common_voxels": count_delta,
        "mean_features_on_common_voxels": feature_difference(old_f, new_f),
        "old_mean_feature_stats": tensor_stats(old_features),
        "spconv2_mean_feature_stats": tensor_stats(new_features),
    }


def compare_stages(old_stages, new_stages):
    result = {}
    for name in old_stages:
        old_coords, old_features = sparse_parts(old_stages[name], "old")
        new_coords, new_features = sparse_parts(new_stages[name], "new")
        old_common, new_common, common = align_common_features(
            old_coords, old_features, new_coords, new_features
        )
        result[name] = {
            "coordinates": coordinate_overlap(old_coords, new_coords),
            "common_coordinate_count": len(common),
            "old_features": tensor_stats(old_features),
            "spconv2_features": tensor_stats(new_features),
            "common_coordinate_feature_difference": feature_difference(
                old_common, new_common
            ),
        }
    return result


def parameter_gradient_stats(model):
    groups = {}
    total_sq = 0.0
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        if name.startswith("encoder_layers."):
            parts = name.split(".")
            group = ".".join(parts[:2])
        if parameter.grad is None:
            continue
        norm = float(torch.linalg.vector_norm(parameter.grad.detach().float()))
        total_sq += norm * norm
        groups.setdefault(group, 0.0)
        groups[group] += norm * norm
    return {
        "total_l2_norm": total_sq ** 0.5,
        "by_stage_l2_norm": {key: value ** 0.5 for key, value in groups.items()},
    }


def compare_gradients(old_features, new_features, old_coords, new_coords,
                      old_backbone, new_backbone):
    """分别反传 mean(BEV^2)，比较体素输入梯度和参数梯度规模。"""
    old_backbone.zero_grad(set_to_none=True)
    new_backbone.zero_grad(set_to_none=True)
    old_leaf = old_features.detach().clone().requires_grad_(True)
    new_leaf = new_features.detach().clone().requires_grad_(True)

    _, old_bev, old_sparse_input = run_old_backbone(
        old_leaf, old_coords, old_backbone, retain_input_gradient=True
    )
    old_loss = old_bev.float().square().mean()
    old_loss.backward()
    old_parameter_gradients = parameter_gradient_stats(old_backbone)

    _, new_bev, new_sparse_input = run_new_backbone(
        new_leaf, new_coords, new_backbone, retain_input_gradient=True
    )
    new_loss = new_bev.float().square().mean()
    new_loss.backward()
    new_parameter_gradients = parameter_gradient_stats(new_backbone)

    # 优先读取 SparseConvTensor 内部实际参与卷积的 features.grad；如果实现直接
    # 使用调用方 leaf，则退回 leaf.grad。任一实现仍不提供输入梯度时，参数梯度
    # 比较照常保存，并在 JSON 中把输入梯度标为 unavailable。
    old_input_grad = old_sparse_input.grad
    if old_input_grad is None:
        old_input_grad = old_leaf.grad
    new_input_grad = new_sparse_input.grad
    if new_input_grad is None:
        new_input_grad = new_leaf.grad
    old_grad, new_grad, common = align_common_features(
        old_coords, old_input_grad, new_coords, new_input_grad
    )
    return {
        "objective": "mean(BEV_feature ** 2)",
        "old_objective_value": float(old_loss.detach()),
        "spconv2_objective_value": float(new_loss.detach()),
        "common_voxel_count": len(common),
        "old_voxel_input_gradient": tensor_stats(old_input_grad),
        "spconv2_voxel_input_gradient": tensor_stats(new_input_grad),
        "common_voxel_gradient_difference": (
            feature_difference(old_grad, new_grad)
            if old_input_grad is not None and new_input_grad is not None
            else {
                "comparable": False,
                "reason": "至少一个稀疏卷积实现未保留体素输入梯度；参数梯度仍已统计",
            }
        ),
        "old_parameter_gradients": old_parameter_gradients,
        "spconv2_parameter_gradients": new_parameter_gradients,
    }


def bev_heatmap(bev):
    return bev[0].detach().float().abs().mean(dim=0).cpu().numpy()


def save_visualization(old_coords, new_coords, old_bev, new_bev, cfg, path):
    """显示体素集合以及新旧 BEV 激活和绝对差异的空间分布。"""
    voxel_cfg = cfg.model.encoders.lidar.voxelize
    point_range = list(voxel_cfg.point_cloud_range)
    voxel_size = np.asarray(voxel_cfg.voxel_size)
    mins = np.asarray(point_range[:3])

    old_xyz = old_coords.detach().cpu().numpy()
    new_xyz = new_coords.detach().cpu().numpy()
    old_set = set(map(tuple, old_xyz.tolist()))
    new_set = set(map(tuple, new_xyz.tolist()))

    def centers(items):
        if not items:
            return np.empty((0, 2))
        xyz = np.asarray(list(items), dtype=np.float32)
        return (mins + (xyz + 0.5) * voxel_size)[:, :2]

    common = centers(old_set & new_set)
    old_only = centers(old_set - new_set)
    new_only = centers(new_set - old_set)
    old_map, new_map = bev_heatmap(old_bev), bev_heatmap(new_bev)
    difference = np.abs(old_map - new_map)
    extent = [point_range[0], point_range[3], point_range[1], point_range[4]]

    fig, axes = plt.subplots(1, 4, figsize=(26, 6), constrained_layout=True)
    axes[0].scatter(common[:, 0], common[:, 1], s=0.08, c="0.55", label="common")
    if len(old_only):
        axes[0].scatter(old_only[:, 0], old_only[:, 1], s=0.5, c="tab:blue", label="old only")
    if len(new_only):
        axes[0].scatter(new_only[:, 0], new_only[:, 1], s=0.5, c="tab:orange", label="spconv2 only")
    axes[0].set_title("Initial non-empty voxel sets")
    axes[0].legend(markerscale=8)

    vmax = max(float(old_map.max()), float(new_map.max()), 1e-12)
    for axis, image, title in zip(
        axes[1:],
        (old_map, new_map, difference),
        ("Old BEV mean |feature|", "spconv2 BEV mean |feature|", "BEV absolute difference"),
    ):
        upper = vmax if "difference" not in title else max(float(difference.max()), 1e-12)
        shown = axis.imshow(
            image, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=upper,
            interpolation="nearest", aspect="equal"
        )
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)

    for axis in axes:
        axis.set_xlim(point_range[0], point_range[3])
        axis.set_ylim(point_range[1], point_range[4])
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args, overrides = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("新旧稀疏卷积对比需要 CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config, overrides)
    points, source = load_training_points(cfg, args, device)
    modules = build_modules(cfg, device)
    old_voxelizer, new_voxelizer, old_backbone, new_backbone = modules

    with torch.no_grad():
        old_voxels = voxelize_and_reduce(points, old_voxelizer)
        new_voxels = voxelize_and_reduce(points, new_voxelizer)
        old_stages, old_bev = run_old_backbone(
            old_voxels[0], old_voxels[1], old_backbone
        )
        new_stages, new_bev = run_new_backbone(
            new_voxels[0], new_voxels[1], new_backbone
        )

    bev_difference = feature_difference(old_bev, new_bev)
    result = {
        "source": source,
        "weight_control": (
            "SparseEncoder 随机初始化一次，并将同一状态转换到 SparseEncoderSpconv2"
        ),
        "execution_mode": {
            "voxelizers": "train：使用配置中的 max_voxels[0]",
            "backbones": "eval：冻结 BN running stats，仍保留 backward",
        },
        "voxel_comparison": compare_voxels(old_voxels, new_voxels),
        "backbone_stage_comparison": compare_stages(old_stages, new_stages),
        "final_bev": {
            "old": tensor_stats(old_bev),
            "spconv2": tensor_stats(new_bev),
            "difference": bev_difference,
        },
        "backward": None,
    }

    if not args.skip_backward:
        # 前向统计使用 no_grad 以节省显存；梯度部分重新执行两次独立前向。
        result["backward"] = compare_gradients(
            old_voxels[0], new_voxels[0], old_voxels[1], new_voxels[1],
            old_backbone, new_backbone,
        )

    scene = source["scene_token"] or "scene_unknown"
    frame = source["frame_token"] or f"frame_index_{args.sample_index}"
    output_dir = args.output_dir.expanduser().resolve() / f"{scene}_{frame}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "spconv2_lidar_equivalence.json"
    image_path = output_dir / "spconv2_lidar_equivalence.png"
    save_visualization(
        old_voxels[1], new_voxels[1], old_bev, new_bev, cfg, image_path
    )
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    voxel_summary = result["voxel_comparison"]["coordinates"]
    print("体素坐标 Jaccard: %.8f" % voxel_summary["jaccard"])
    print("公共体素点数一致率: %s" % result["voxel_comparison"]["point_counts_on_common_voxels"]["equal_fraction"])
    print("最终 BEV relative L2: %s" % bev_difference.get("relative_l2"))
    if result["backward"] is not None:
        print(
            "公共体素梯度 relative L2: %s"
            % result["backward"]["common_voxel_gradient_difference"].get("relative_l2")
        )
    print(f"JSON: {json_path}")
    print(f"可视化: {image_path}")


if __name__ == "__main__":
    main()
