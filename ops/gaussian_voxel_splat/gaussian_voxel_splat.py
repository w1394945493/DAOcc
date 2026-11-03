import torch
from torch import nn

try:
    from . import _C
except ImportError as error:
    _C = None
    _IMPORT_ERROR = error


class _GaussianVoxelSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, points_int, means, means_int, opacity, features,
                radii, precision, size_x, size_y, size_z, sigma_limit,
                normalized):
        if _C is None:
            raise ImportError(
                "gaussian_voxel_splat尚未编译，请在ops/gaussian_voxel_splat下执行 "
                "python setup.py build_ext --inplace"
            ) from _IMPORT_ERROR
        with torch.cuda.device(means.device):
            torch.cuda.synchronize(means.device)
            rendered, out, weight_sum, geom, binning, image = _C.forward(
                points, points_int, means, means_int, opacity, features,
                radii, precision, size_x, size_y, size_z, sigma_limit,
                normalized)
            torch.cuda.synchronize(means.device)
        ctx.rendered = rendered
        ctx.grid_size = (size_x, size_y, size_z)
        ctx.sigma_limit = sigma_limit
        ctx.normalized = normalized
        ctx.save_for_backward(geom, binning, image, means, points, points_int,
                              precision, opacity, features, out, weight_sum)
        # weight_sum仅用于覆盖率统计；当前接口不把它作为可求导的模型输出。
        ctx.mark_non_differentiable(weight_sum)
        return out, weight_sum

    @staticmethod
    def backward(ctx, out_grad, _weight_sum_grad):
        geom, binning, image, means, points, points_int, precision, opacity, \
            features, out, weight_sum = ctx.saved_tensors
        size_x, size_y, size_z = ctx.grid_size
        with torch.cuda.device(means.device):
            torch.cuda.synchronize(means.device)
            means_grad, opacity_grad, features_grad, precision_grad = _C.backward(
                geom, binning, image, size_x, size_y, size_z, ctx.rendered,
                means, points, points_int, precision, opacity, features, out,
                weight_sum, out_grad.contiguous(), ctx.sigma_limit,
                ctx.normalized)
            torch.cuda.synchronize(means.device)
        return (None, None, means_grad, None, opacity_grad, features_grad,
                None, precision_grad, None, None, None, None, None)


def _build_grid(pc_range, grid_size, device):
    """生成[X,Y,Z]整数索引和对应物理体素中心。"""
    size_x, size_y, size_z = [int(value) for value in grid_size]
    minimum = torch.as_tensor(pc_range[:3], dtype=torch.float32, device=device)
    maximum = torch.as_tensor(pc_range[3:], dtype=torch.float32, device=device)
    step = (maximum - minimum) / torch.tensor(
        [size_x, size_y, size_z], dtype=torch.float32, device=device)
    ix = torch.arange(size_x, dtype=torch.int32, device=device)
    iy = torch.arange(size_y, dtype=torch.int32, device=device)
    iz = torch.arange(size_z, dtype=torch.int32, device=device)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    points_int = torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3).contiguous()
    points = minimum + (points_int.float() + 0.5) * step
    return points.contiguous(), points_int, minimum, step


def gaussian_voxel_splat(centers, scales, rotations, opacities, features,
                          pc_range, grid_size, sigma_limit=3.0,
                          accumulation="normalized"):
    """将3D Gaussian特征聚合到[B=1,C,Z,Y,X]规则体素。

    rotations为[G,3,3]旋转矩阵；返回值保留完整3D高度维。当前CUDA版本固定
    支持FP32、128维特征和单样本，离散邻域半径不参与求导。
    """
    tensors = (centers, scales, rotations, opacities, features)
    if any(not value.is_cuda for value in tensors):
        raise ValueError("gaussian_voxel_splat的全部输入必须位于CUDA")
    if any(value.dtype != torch.float32 for value in tensors):
        raise TypeError("当前gaussian_voxel_splat仅支持FP32")
    if features.ndim != 2 or features.shape[1] != 128:
        raise ValueError(f"features必须为[G,128]，实际为{tuple(features.shape)}")
    if accumulation not in ("normalized", "sum"):
        raise ValueError(f"不支持的accumulation：{accumulation}")
    opacity = opacities.reshape(-1).contiguous()
    points, points_int, minimum, step = _build_grid(
        pc_range, grid_size, centers.device)
    size_x, size_y, size_z = [int(value) for value in grid_size]

    # Sigma=R diag(s^2) R^T；CUDA使用Sigma^{-1}的六个独立元素。
    scales_safe = scales.clamp_min(1e-4)
    covariance = rotations @ torch.diag_embed(scales_safe.square()) @ rotations.transpose(-1, -2)
    precision_matrix = torch.linalg.inv(covariance)
    precision = precision_matrix[:, [0, 1, 2, 0, 1, 0],
                                 [0, 1, 2, 1, 2, 2]].contiguous()

    # 包围盒半径只决定离散候选集合；连续权重仍由precision和sigma_limit判断。
    world_std = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
    radii = torch.ceil(sigma_limit * world_std.detach() / step).to(torch.int32)
    radii = torch.minimum(radii, torch.tensor(
        [size_x - 1, size_y - 1, size_z - 1], device=centers.device,
        dtype=torch.int32)).contiguous()
    means_int = torch.floor((centers.detach() - minimum) / step).to(torch.int32)
    means_int = torch.maximum(means_int, torch.zeros_like(means_int))
    means_int = torch.minimum(means_int, torch.tensor(
        [size_x - 1, size_y - 1, size_z - 1], device=centers.device,
        dtype=torch.int32)).contiguous()

    flat, weight_sum = _GaussianVoxelSplat.apply(
        points, points_int, centers.contiguous(), means_int, opacity,
        features.contiguous(), radii, precision, size_x, size_y, size_z,
        float(sigma_limit), accumulation == "normalized")
    # flat顺序为[X,Y,Z,C]，恢复并转换成标准体素排列[B,C,Z,Y,X]。
    volume = flat.reshape(size_x, size_y, size_z, 128).permute(
        3, 2, 1, 0).unsqueeze(0).contiguous()
    weight_volume = weight_sum.reshape(size_x, size_y, size_z).permute(
        2, 1, 0).unsqueeze(0).unsqueeze(0).contiguous()
    return volume, weight_volume


class GaussianVoxelSplat(nn.Module):
    def __init__(self, pc_range, grid_size, sigma_limit=3.0,
                 accumulation="normalized"):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.grid_size = tuple(int(value) for value in grid_size)
        self.sigma_limit = float(sigma_limit)
        self.accumulation = accumulation

    def forward(self, centers, scales, rotations, opacities, features):
        return gaussian_voxel_splat(
            centers, scales, rotations, opacities, features, self.pc_range,
            self.grid_size, self.sigma_limit, self.accumulation)
