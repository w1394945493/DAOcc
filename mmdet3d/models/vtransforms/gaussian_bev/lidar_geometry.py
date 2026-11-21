"""Build Gaussian centres and image sampling coordinates from LiDAR points."""
import torch
from torch import nn
from torch.nn import functional as F


class LidarGaussianGeometry(nn.Module):
    def __init__(self, image_size, gaussian_std=(0.3, 0.3, 0.3),
                 time_dimension=4, current_time_epsilon=1e-6):
        super().__init__()
        if len(gaussian_std) != 3 or any(v <= 0 for v in gaussian_std):
            raise ValueError("gaussian_std must contain three positive values")
        self.image_size = tuple(image_size)
        self.time_dimension = time_dimension
        self.current_time_epsilon = current_time_epsilon
        variance = torch.as_tensor(gaussian_std, dtype=torch.float32).square()
        # Upper-triangular order consumed by the Gaussian rasterizer.
        covariance = torch.stack((variance[0], variance.new_zeros(()),
                                  variance.new_zeros(()), variance[1],
                                  variance.new_zeros(()), variance[2]))
        self.register_buffer("covariance", covariance, persistent=False)

    def forward(self, image_features, image_opacity, points, lidar2image,
                img_aug_matrix, lidar_aug_matrix):
        """Return padded per-point Gaussian tensors for a batch."""
        _, num_cameras, channels = image_features.shape[:3]
        # points是长度为B的Python列表，而不是[B,P,C]定长张量；其中第b项形状为
        # [P_b,C]，所以同一batch内不同样本的原始LiDAR点数P_b可以不同。
        # 在完成每个样本的投影和特征采样之前，先用列表保存不定长结果。
        sample_features, sample_means, sample_cov, sample_opacity = [], [], [], []

        # 按batch逐样本处理，每个样本独立使用自己的点数、标定矩阵和数据增强矩阵。
        for batch_index, point_tensor in enumerate(points):
            xyz_aug = point_tensor[:, :3].float()
            count = xyz_aug.shape[0]
            if count:
                inverse_aug = torch.linalg.inv(lidar_aug_matrix[batch_index].float())
                homogeneous = torch.cat((xyz_aug, xyz_aug.new_ones(count, 1)), 1)
                xyz_raw = (inverse_aug @ homogeneous.t()).t()[:, :3]
                projection_points = torch.cat(
                    (xyz_raw, point_tensor.new_ones(count, 1).float()), 1)
                camera = torch.matmul(
                    lidar2image[batch_index, :, :3, :].float(), projection_points.t())
                depth = camera[:, 2]
                image = camera / depth.clamp_min(1e-5).unsqueeze(1)
                aug = img_aug_matrix[batch_index].float()
                image = torch.matmul(aug[:, :3, :3], image) + aug[:, :3, 3:4]
                uv = image[:, :2].transpose(1, 2)
                height, width = self.image_size
                visible = ((depth > 1e-5) & (uv[..., 0] >= 0) &
                           (uv[..., 0] < width) & (uv[..., 1] >= 0) &
                           (uv[..., 1] < height))
                if point_tensor.shape[1] > self.time_dimension:
                    time_lag = point_tensor[:, self.time_dimension].float().abs()
                    is_current = time_lag <= self.current_time_epsilon
                else:
                    # 单帧点云没有时间维时，全部按当前帧点处理。
                    is_current = torch.ones(count, dtype=torch.bool,
                                            device=xyz_aug.device)

                feature_height, feature_width = image_features.shape[-2:]
                feature_count = feature_height * feature_width
                feature_maps = image_features[batch_index].permute(0, 2, 3, 1).reshape(
                    num_cameras, feature_count, channels)
                opacity_maps = image_opacity[batch_index].permute(0, 2, 3, 1).reshape(
                    num_cameras, feature_count, 1)
                selected_features, selected_means, selected_opacity = [], [], []

                # 每个相机特征格至多选择一个LiDAR点：若该格存在当前帧点，只在
                # 当前帧点中选择相机深度最近者；否则从历史sweep点中选择最近者。
                for camera_index in range(num_cameras):
                    camera_visible = visible[camera_index]
                    if not camera_visible.any():
                        continue
                    point_indices = torch.nonzero(camera_visible, as_tuple=False)[:, 0]
                    camera_uv = uv[camera_index, point_indices]
                    feature_x = torch.floor(
                        camera_uv[:, 0] * feature_width / width).long().clamp(0, feature_width - 1)
                    feature_y = torch.floor(
                        camera_uv[:, 1] * feature_height / height).long().clamp(0, feature_height - 1)
                    cell = feature_y * feature_width + feature_x
                    camera_depth = depth[camera_index, point_indices]

                    # 历史点增加一个大于当前相机最大深度的优先级偏移，因此只要某格
                    # 存在当前帧点，历史点就不可能成为该格的最小score。
                    history_offset = camera_depth.max() + 1.0
                    score = camera_depth + (~is_current[point_indices]).to(
                        camera_depth.dtype) * history_offset
                    minimum = score.new_full((feature_count,), float("inf"))
                    minimum.scatter_reduce_(0, cell, score, reduce="amin", include_self=True)

                    # 深度完全相同（例如场景起始处pad_empty_sweeps复制当前帧）时，
                    # 再以原点索引取最小者，确保每个特征格严格只保留一个点。
                    candidate = score == minimum[cell]
                    candidate_points = point_indices[candidate]
                    candidate_cells = cell[candidate]
                    winner = torch.full((feature_count,), count, dtype=torch.long,
                                        device=xyz_aug.device)
                    winner.scatter_reduce_(0, candidate_cells, candidate_points,
                                           reduce="amin", include_self=True)
                    occupied_cells = torch.nonzero(winner < count, as_tuple=False)[:, 0]
                    chosen_points = winner[occupied_cells]
                    selected_features.append(feature_maps[camera_index, occupied_cells])
                    selected_opacity.append(opacity_maps[camera_index, occupied_cells])
                    selected_means.append(xyz_aug[chosen_points])

                if selected_means:
                    feat = torch.cat(selected_features, dim=0)
                    alpha = torch.cat(selected_opacity, dim=0)
                    xyz_aug = torch.cat(selected_means, dim=0)
                else:
                    feat = image_features.new_empty(0, channels)
                    alpha = image_opacity.new_empty(0, 1)
                    xyz_aug = xyz_aug.new_empty(0, 3)
                count = xyz_aug.shape[0]
                covariance = self.covariance.to(xyz_aug).expand(count, -1)
            else:
                feat = image_features.new_empty(0, channels)
                alpha = image_opacity.new_empty(0, 1)
                covariance = xyz_aug.new_empty(0, 6)

            sample_features.append(feat)
            sample_means.append(xyz_aug)
            sample_cov.append(covariance)
            sample_opacity.append(alpha)

        # torch.stack要求每个样本形状一致，因此取当前batch内最大的有效点数P_max，
        # 将其余样本补齐到P_max。这里只在当前batch内部补齐，不要求整个数据集点数一致。
        max_points = max((value.shape[0] for value in sample_means), default=0)
        padded_features, padded_means, padded_cov, padded_opacity = [], [], [], []
        for feat, means, covariance, alpha in zip(
                sample_features, sample_means, sample_cov, sample_opacity):
            pad = max_points - means.shape[0]
            # F.pad默认使用0：尤其是补齐高斯的opacity为0，renderer会在调用CUDA
            # 光栅化前通过opacity阈值将它们过滤，因此padding不会写入BEV特征。
            padded_features.append(F.pad(feat, (0, 0, 0, pad)))
            padded_means.append(F.pad(means, (0, 0, 0, pad)))
            padded_cov.append(F.pad(covariance, (0, 0, 0, pad)))
            padded_opacity.append(F.pad(alpha, (0, 0, 0, pad)))

        # 补齐后得到features[B,P_max,C]、means[B,P_max,3]、cov[B,P_max,6]
        # 和opacity[B,P_max,1]，从而可以按标准batch张量传给renderer。
        return (torch.stack(padded_features), torch.stack(padded_means),
                torch.stack(padded_cov), torch.stack(padded_opacity))
