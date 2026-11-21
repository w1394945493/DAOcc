"""Gaussian feature splatting with DAOcc's [X,Y] BEV layout."""
import torch
from torch import nn


class GaussianBEVRenderer(nn.Module):
    def __init__(self, xbound, ybound, channels=128, opacity_filter=.01):
        super().__init__()
        if channels != 128:
            raise ValueError("The reference CUDA rasterizer is compiled for 128 channels")
        if not 0 <= opacity_filter < 1:
            raise ValueError("opacity_filter must be in [0,1)")
        self.channels = channels
        self.threshold = opacity_filter
        sizes = []
        for lo, hi, step in (xbound, ybound):
            if step <= 0 or hi <= lo:
                raise ValueError("Invalid BEV bounds")
            count = (hi-lo)/step
            if abs(count-round(count)) > 1e-5:
                raise ValueError("BEV bounds must contain an integer number of cells")
            sizes.append(round(count))
        self.height, self.width = sizes  # tensor rows X, columns Y
        sx, sy = 1./xbound[2], 1./ybound[2]
        view = torch.zeros(4, 4)
        view[0, 1], view[1, 0] = sy, sx
        proj = torch.zeros(4, 4)
        proj[0, 1], proj[1, 0] = sy, sx
        proj[0, 3] = -ybound[0]*sy-.5
        proj[1, 3] = -xbound[0]*sx-.5
        proj[2, 3] = proj[3, 3] = 1.
        self.register_buffer("view", view, persistent=False)
        self.register_buffer("projection", proj, persistent=False)

    def forward(self, features, means, covariances, opacities):
        if not features.is_cuda:
            raise RuntimeError("GaussianBEVTransform rendering requires CUDA")
        try:
            from mmdet3d.ops.gaussian_rasterization.rasterize import (
                GaussianRasterizationSettings, GaussianRasterizer)
        except ImportError as exc:
            raise ImportError("Build the Gaussian CUDA extension from DAOcc with: "
                              "python mmdet3d/ops/gaussian_rasterization/setup.py build_ext --inplace") from exc
        settings = GaussianRasterizationSettings(
            image_height=self.height, image_width=self.width,
            tanfovx=0., tanfovy=0., bg=features.new_zeros(self.channels),
            scale_modifier=1., viewmatrix=self.view.float(),
            projmatrix=self.projection.float(), sh_degree=0,
            campos=features.new_zeros(3), prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer()
        rasterizer.set_raster_settings(settings)
        outputs = []
        # The upstream kernel allocates scratch on the current CUDA device.
        with torch.cuda.device(features.device):
            # CUDA rasterizer一次处理一个样本。这里遍历batch，允许每个样本通过下方
            # opacity mask恢复各自真实的有效高斯数量，最后再将BEV结果stack成batch。
            for i in range(features.shape[0]):
                # lidar_geometry为适配不同LiDAR点数而补齐的高斯，其opacity为0；
                # 此mask同时移除padding和低于配置阈值的真实高斯。
                mask = opacities[i, :, 0] > self.threshold
                if not mask.any():
                    # Keep all branches in autograd even if hard filtering removes every Gaussian.
                    zero = sum(t[i].sum() * 0 for t in (features, means, covariances, opacities))
                    outputs.append(features.new_zeros(self.channels, self.height, self.width) + zero)
                    continue
                bev, _ = rasterizer(
                    means3D=means[i][mask].contiguous(), means2D=None, shs=None,
                    colors_precomp=features[i][mask].contiguous(),
                    opacities=opacities[i][mask].contiguous(), scales=None, rotations=None,
                    cov3D_precomp=covariances[i][mask].contiguous())
                outputs.append(bev)
        return torch.stack(outputs)
