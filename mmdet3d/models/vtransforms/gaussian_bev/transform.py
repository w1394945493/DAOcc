"""Gaussian BEV transform with depth-derived or LiDAR-derived centres."""
import torch
from mmcv.runner import BaseModule
from mmdet3d.models.builder import VTRANSFORMS
from .predictor import PrepareChannel, PrepareLidarGaussianChannel
from .geometry import GaussianGeometry
from .lidar_geometry import LidarGaussianGeometry
from .renderer import GaussianBEVRenderer


@VTRANSFORMS.register_module()
class GaussianBEVTransform(BaseModule):
    def __init__(self, in_channels=256, out_channels=128,
                 image_size=(256, 704), feature_size=(32, 88),
                 xbound=(-54., 54., .6), ybound=(-54., 54., .6),
                 depth_num=64, depth_start=1., depth_max=61.,
                 error_tolerance=.5, opacity_filter=.01, interm_channels=256,
                 center_source="depth", gaussian_std=(0.3, 0.3, 0.3),
                 lidar_time_dimension=4, current_time_epsilon=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = tuple(feature_size)
        if center_source not in ("depth", "lidar"):
            raise ValueError("center_source must be 'depth' or 'lidar'")
        self.center_source = center_source
        if center_source == "depth":
            self.predictor = PrepareChannel(in_channels=[in_channels],
                                            interm_c=interm_channels,
                                            out_c=out_channels, depth_num=depth_num)
            self.geometry = GaussianGeometry(image_size, feature_size, depth_num,
                                             depth_start, depth_max, error_tolerance)
        else:
            self.predictor = PrepareLidarGaussianChannel(
                in_channels=[in_channels], interm_c=interm_channels,
                out_c=out_channels)
            self.geometry = LidarGaussianGeometry(
                image_size, gaussian_std, lidar_time_dimension,
                current_time_epsilon)
        self.renderer = GaussianBEVRenderer(xbound, ybound, out_channels, opacity_filter)
        self.fp16_enabled = False

    def forward(self, x, points, camera2ego, lidar2ego, lidar2camera,
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas, **kwargs):
        b, n, c, h, w = x.shape
        if c != self.in_channels or (h, w) != self.feature_size:
            raise ValueError("Unexpected image feature shape: {}".format(tuple(x.shape)))
        # Reference rasterizer consumes float pointers. Explicitly disable AMP even
        # when the enclosing DAOcc training loop uses native autocast.
        with torch.cuda.amp.autocast(enabled=False):
            flat = x.float().reshape(b*n, c, h, w)
            if self.center_source == "depth":
                features, depth, opacity = self.predictor(flat)
                means, cov = self.geometry(
                    depth, camera_intrinsics.float(), camera2lidar.float(),
                    img_aug_matrix.float(), lidar_aug_matrix.float())
                features = features.reshape(b, n, self.out_channels, h, w).permute(0,1,3,4,2).reshape(b,-1,self.out_channels)
                opacity = opacity.reshape(b,n,1,h,w).permute(0,1,3,4,2).reshape(b,-1,1)
            else:
                features, opacity = self.predictor(flat)
                features = features.reshape(b, n, self.out_channels, h, w)
                opacity = opacity.reshape(b, n, 1, h, w)
                features, means, cov, opacity = self.geometry(
                    features, opacity, points, lidar2image.float(),
                    img_aug_matrix.float(), lidar_aug_matrix.float())
            return self.renderer(features, means, cov, opacity)
