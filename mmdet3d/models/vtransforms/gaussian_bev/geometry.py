"""GaussianLSS depth moments in DAOcc's augmented LiDAR coordinates."""
import torch
from torch import nn


class GaussianGeometry(nn.Module):
    def __init__(self, image_size, feature_size, depth_num=64,
                 depth_start=1., depth_max=61., error_tolerance=0.5):
        super().__init__()
        if depth_num < 1 or depth_max <= depth_start or depth_start <= 0:
            raise ValueError("Invalid depth bins")
        if error_tolerance <= 0:
            raise ValueError("error_tolerance must be positive")
        self.feature_size = tuple(feature_size)
        self.depth_num = depth_num
        self.error_tolerance = error_tolerance
        # Same uniform bin centers and endpoint pixel sampling as GaussianLSS.
        edges = torch.linspace(depth_start, depth_max, depth_num + 1)
        bins = (edges[:-1] + edges[1:]) * .5
        h, w = feature_size
        ys = torch.linspace(0, image_size[0], h)
        xs = torch.linspace(0, image_size[1], w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack((xx, yy), -1)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("pixels", pixels, persistent=False)

    @torch.no_grad()
    def candidate_points(self, camera_intrinsics, camera2lidar,
                         img_aug_matrix, lidar_aug_matrix):
        b, n = camera2lidar.shape[:2]
        d = self.depth_num
        h, w = self.feature_size
        pixels = self.pixels.float().view(1, 1, 1, h, w, 2).expand(b, n, d, h, w, 2)
        depths = self.bins.float().view(1, 1, d, 1, 1, 1).expand(b, n, d, h, w, 1)
        # DAOcc stores image post-transform separately from camera calibration.
        points = torch.cat((pixels, depths), -1)
        points = points - img_aug_matrix[..., :3, 3].view(b, n, 1, 1, 1, 3)
        inv_post = torch.linalg.inv(img_aug_matrix[..., :3, :3])
        points = (inv_post.view(b, n, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)).squeeze(-1)
        points = torch.cat((points[..., :2] * points[..., 2:3], points[..., 2:3]), -1)
        rotation = camera2lidar[..., :3, :3] @ torch.linalg.inv(camera_intrinsics[..., :3, :3])
        points = (rotation.view(b, n, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)).squeeze(-1)
        points = points + camera2lidar[..., :3, 3].view(b, n, 1, 1, 1, 3)
        points = (lidar_aug_matrix[:, :3, :3].view(b, 1, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)).squeeze(-1)
        return points + lidar_aug_matrix[:, :3, 3].view(b, 1, 1, 1, 1, 3)

    def forward(self, depth_logits, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix):
        b, n = camera2lidar.shape[:2]
        h, w = self.feature_size
        coords = self.candidate_points(camera_intrinsics, camera2lidar,
                                       img_aug_matrix, lidar_aug_matrix)
        probability = depth_logits.reshape(b, n, self.depth_num, h, w).softmax(2)
        means = (probability.unsqueeze(-1) * coords).sum(2)
        delta = means.unsqueeze(2) - coords
        cov = (probability[..., None, None] *
               (delta.unsqueeze(-1) @ delta.unsqueeze(-2))).sum(2)
        cov = cov * (self.error_tolerance ** 2 / 9.)
        # Upper triangular order required by the original CUDA kernel.
        cov = cov.flatten(-2)[..., [0, 1, 2, 4, 5, 8]]
        return means.reshape(b, -1, 3), cov.reshape(b, -1, 6)
