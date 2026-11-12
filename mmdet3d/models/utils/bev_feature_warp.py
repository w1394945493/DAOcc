"""StreamOcc-style inverse motion warping adapted to DAOcc's 2D [B,C,X,Y] BEV."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BEVFeatureWarp(nn.Module):
    """Warp a detached history into the current augmented LiDAR BEV grid.

    Points lie on ``z=plane_z`` in the CURRENT AUGMENTED LiDAR frame. The
    complete 4x4 transform is applied, then historical XY is sampled. Height
    has already been collapsed into channels: this is a planar approximation,
    not StreamOcc's 3D voxel resampling or dynamic-object motion compensation.

    DAOcc tensor axes are X then Y. Thus grid_sample's (column, row) is (Y, X).
    Range bounds describe cell edges; sample positions are cell centers. Like
    StreamAgg, sampling uses align_corners=True, with normalization to the
    first/last CELL CENTERS rather than to the physical range edges.
    No parameters or persistent buffers are added to the model state_dict.
    """

    def __init__(self, point_cloud_range, plane_z=0.0, mask_outside=True):
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError('BEV warp expects [xmin,ymin,zmin,xmax,ymax,zmax]')
        self.point_cloud_range = tuple(float(v) for v in point_cloud_range)
        self.plane_z = float(plane_z)
        #* (11.12) 仅缓存模式保持旧掩码行为；融合模式使用 StreamOcc 的零填充插值。
        self.mask_outside = mask_outside
        if not all(math.isfinite(v) for v in (*self.point_cloud_range, self.plane_z)):
            raise ValueError('BEV range and plane_z must be finite')
        if any(self.point_cloud_range[i + 3] <= self.point_cloud_range[i] for i in range(3)):
            raise ValueError('BEV range maxima must exceed minima')

    @staticmethod
    def _matrices(metas, key, device):
        matrices = torch.tensor([meta[key] for meta in metas], device=device, dtype=torch.float64)
        if matrices.shape != (len(metas), 4, 4) or not torch.isfinite(matrices).all():
            raise ValueError(f'Invalid cached/current {key}')
        bottom = matrices.new_tensor([0, 0, 0, 1]).expand(len(metas), -1)
        if not torch.allclose(matrices[:, 3], bottom):
            raise ValueError(f'{key} must be homogeneous transforms')
        return matrices

    @torch.no_grad()
    def forward(self, history_bev, history_metas, current_metas, history_valid):
        if history_bev.ndim != 4:
            raise ValueError('BEV warp expects [B,C,X,Y]')
        batch, _, nx, ny = history_bev.shape
        if nx < 2 or ny < 2:
            raise ValueError('BEV warp requires at least two cells along each axis')
        if len(history_metas) != batch or len(current_metas) != batch:
            raise ValueError('BEV warp metadata batch mismatch')
        device = history_bev.device
        valid = torch.as_tensor(history_valid, device=device, dtype=torch.bool)
        if valid.shape != (batch,):
            raise ValueError('BEV warp validity must have shape [B]')

        #* (11.12) 参照 StreamAgg 的逆向采样，补入 DAOcc 的 LiDAR 外参及两帧增强。
        # T = A_prev @ inv(G_prev) @ G_cur @ inv(A_cur), G = ego2global @ lidar2ego。
        # 位姿组合用 FP64；网格与插值用 FP32，避免混合精度下的坐标量化。
        with torch.cuda.amp.autocast(enabled=False):
            transform = torch.eye(4, device=device, dtype=torch.float64).repeat(batch, 1, 1)
            indices = valid.nonzero(as_tuple=False).flatten()
            if indices.numel():
                slots = indices.tolist()
                old = [history_metas[i] for i in slots]
                cur = [current_metas[i] for i in slots]
                old_global = self._matrices(old, 'ego2global', device) @ self._matrices(old, 'lidar2ego', device)
                cur_global = self._matrices(cur, 'ego2global', device) @ self._matrices(cur, 'lidar2ego', device)
                old_aug = self._matrices(old, 'lidar_aug_matrix', device)
                cur_aug = self._matrices(cur, 'lidar_aug_matrix', device)
                transform[indices] = old_aug @ torch.linalg.solve(old_global, cur_global) @ torch.linalg.inv(cur_aug)
            if not torch.isfinite(transform).all():
                raise ValueError('Non-finite BEV alignment transform')

            xmin, ymin, _, xmax, ymax, _ = self.point_cloud_range
            dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
            xs = xmin + (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * dx
            ys = ymin + (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * dy
            xx, yy = torch.meshgrid(xs, ys, indexing='ij')
            points = torch.stack((xx, yy, torch.full_like(xx, self.plane_z), torch.ones_like(xx)), -1)
            previous = torch.einsum('bij,xyj->bxyi', transform.float(), points)
            ix = (previous[..., 0] - (xmin + 0.5 * dx)) / dx
            iy = (previous[..., 1] - (ymin + 0.5 * dy)) / dy
            # Tensor H=X,W=Y: grid coordinates must be (normalized Y, normalized X).
            grid = torch.stack((2 * iy / (ny - 1) - 1, 2 * ix / (nx - 1) - 1), dim=-1)
            if not torch.isfinite(grid).all():
                raise ValueError('Non-finite BEV sampling grid')
            # Full interpolation support only; ignore tiny FP32 roundoff at edges.
            in_bounds = (grid.abs() <= 1 + 1e-5).all(dim=-1)
            pixel_valid = in_bounds & valid[:, None, None]
            # Snap roundoff at boundary centers, while retaining truly out-of-range coordinates.
            grid = torch.where(in_bounds[..., None], grid.clamp(-1, 1), grid)
            aligned = F.grid_sample(history_bev.detach().float(), grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
            if self.mask_outside:
                aligned.masked_fill_(~pixel_valid[:, None], 0)
            else:
                aligned.masked_fill_(~valid[:, None, None, None], 0)
            aligned = aligned.to(history_bev.dtype)
        return dict(bev=aligned, valid_mask=pixel_valid, current_to_history=transform, grid=grid)
