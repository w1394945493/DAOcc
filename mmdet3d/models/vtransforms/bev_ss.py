from typing import Tuple, List

import torch
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from torch import nn
import torch.nn.functional as F
import einops
from mmdet3d.models.builder import VTRANSFORMS
from ..utils.reference_poinst import get_reference_points

#* ================== 两个版本的共同流程与差异 ==================
#* 总览：两种实现都从规则 3D 网格出发，投影到图像采样，再把高度层拼接到通道维得到 BEV。
#* 均不预测 LSS 式逐像素深度概率；同一射线上的不同网格点可能采样到相同图像特征。
# BEVTransform：逐相机采样，同一网格点的有效相机特征取平均。
# BEVTransformV2：每点只选第一个有效相机，把相机编号作为 grid_sample 的第三个坐标。
# 两版均只调用一次 grid_sample；V2 的优化重点是减少重叠区域重复采样和逐相机回填聚合。
# V2 类说明注明参考 SparseBEV 优化采样；仓库保留原版供仍使用 BEVTransform 的配置选择。
#* V2 改变了多视角聚合策略和 align_corners，不是保证数值等价的替换；速度与精度需实测。
# 记号：B 为 batch，N 为相机数，C 为特征通道，H/W 为图像特征尺寸，NP 为网格点总数。

__all__ = ["BEVTransform", "BEVTransformV2"]


@VTRANSFORMS.register_module()
class BEVTransform(BaseModule):
    r"""
    Please refer to the paper DAOcc['https://arxiv.org/abs/2409.19972'].

    Args:
        x, y, z: Range in three directions.
        xs, ys, zs: Resolution in three directions.
        input_size (tuple(int)): Size of input images in format of (height, width).
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        top_type (str): Coordinate system in which the task is labeled, 'lidar' or 'ego',
                        eg: 'lidar' for SurroundOcc and 'ego' for Occ3D (camera only)
    """

    def __init__(
            self,
            x: List[float], y: List[float], z: List[float],
            xs: int, ys: int, zs: int,
            input_size: List[int],
            in_channels: int = 256,
            out_channels: int = 128,
            top_type: str = 'lidar',
            down_sample: bool = False,
            down_sample_scale: int = 2,
            down_sample_channels: List[int] = [128*10, 64*10, 32*10, 16*10]
    ):
        super(BEVTransform, self).__init__()

        #* ================== 1. 创建三维参考网格 ==================
        # 初始化：在目标坐标系内建立固定的 3D 体素中心网格。
        # xs/ys/zs 为各轴网格数量；NP = xs * ys * zs，zs 是高度层数，不是图像深度 bin。
        self.pc_range = [x[0], y[0], z[0], x[1], y[1], z[1]]
        self.volume_size = [int(s) for s in [xs, ys, zs]]
        ref_3d = get_reference_points(self.pc_range[0], self.pc_range[3],
                                      self.pc_range[1], self.pc_range[4],
                                      self.pc_range[2], self.pc_range[5],
                                      self.volume_size[0], self.volume_size[1], self.volume_size[2])
        # ref_3d: [NP, 3]，随模型迁移设备，但不是可学习参数。
        self.register_buffer('ref_3d', ref_3d)
        self.transfer_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.initial_flag = True
        self.input_size = input_size
        self.top_type = top_type

        #* ================== 2. 配置可选 BEV 卷积 ==================
        # 可选 BEV 卷积：输入通道应为 zs * out_channels。
        # scale=2 同时降低空间分辨率；scale=1 保持分辨率，只做卷积变换。
        if down_sample:
            assert down_sample_scale in [1, 2], 'only support 1x or 2x down sampling!'
            if down_sample_scale == 2:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True),
                    nn.Conv2d(
                        down_sample_channels[1],
                        down_sample_channels[2],
                        3,
                        stride=down_sample_scale,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(down_sample_channels[2]),
                    nn.ReLU(True),
                    nn.Conv2d(down_sample_channels[2], down_sample_channels[3], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[3]),
                    nn.ReLU(True),
                )
            else:  # down_sample_scale == 1
                assert len(down_sample_channels) == 2, "the len of 'down_sample_channels: List[int]' should be 2!"
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True)
                )
        else:
            self.down_sample = None
        self.fp16_enabled = False

    @force_fp32()
    def forward(self,
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs
                ):
        """Transform image-view feature into bird-eye-view feature.

        Args:
        Returns:
        """

        #* ================== 1. 准备逐相机图像特征 ==================
        # 准备图像特征：输入 [B,N,C,H,W]，合并 batch/相机维，逐相机做 1x1 通道映射。
        # 原版始终执行 transfer_conv，即使输入输出通道相同。
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.transfer_conv(x)
        _, C, _, _ = x.shape

        #* ================== 2. 确定目标坐标系 ==================
        # 确定投影的目标坐标系，camera2sensor 表示相机到该坐标系的变换。
        # ego：经 global 对齐到第一个相机对应的关键帧 ego；lidar：直接使用 camera2lidar。
        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)  # (B, 1, 4, 4)
            global2keyego = torch.inverse(keyego2global.double())  # (B, 1, 4, 4)
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()  # (B, N_views, 4, 4)
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError
        # 将固定 3D 网格投影到图像；这里只计算采样坐标和有效掩码，还未读取图像特征。
        reference_points_img, volume_mask = self.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
                                                                img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                                lidar_aug_matrix)

        #* ================== 3. 按相机整理有效采样点 ==================
        # 按相机收集有效网格点，并补齐到 max_len，减少无效位置的采样开销。
        # 返回坐标 [N,B,NP,1,2]；每个体素只有一个中心点，最后两维为 1 和 (u,v)。
        num_cams, bs, num_voxels, num_points_in_voxel, _ = reference_points_img.shape
        indexes = [[] for _ in range(bs)]
        for i, mask_per_img in enumerate(volume_mask):
            for j in range(bs):
                assert mask_per_img[j].shape[-1] == 1
                index_query_per_img = mask_per_img[j].squeeze(-1).nonzero().squeeze(-1)
                indexes[j].append(index_query_per_img)
        all_indexes = [item for sublist in indexes for item in sublist]
        max_len = max([len(each) for each in all_indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        reference_points_rebatch = reference_points_img.new_zeros([bs, num_cams, max_len, num_points_in_voxel, 2])
        for j in range(bs):
            indexes_per_batch = indexes[j]
            for ii, reference_points_per_img in enumerate(reference_points_img):
                index_query_per_img = indexes_per_batch[ii]
                reference_points_rebatch[j, ii, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]
        reference_points_rebatch = reference_points_rebatch.view(bs * num_cams, max_len, num_points_in_voxel, 2)
        reference_points_rebatch = 2 * reference_points_rebatch - 1

        #* ================== 4. 二维图像特征采样 ==================
        # 将所有相机合并到 batch 维，一次 grid_sample 完成采样，并非逐相机调用该算子。
        # 对每台相机的二维特征做双线性插值；采样坐标已从 [0,1] 转成 [-1,1]。
        # 原版 align_corners=False，与 V2 的 True 有差异，两者不保证数值等价。
        # x: bn, c, h, w
        # reference_points_rebatch: bn, max_len, num_points_in_voxel, uv/xy/wh
        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        # bn, c, max_len, num_points_in_voxel
        sampling_feats = einops.rearrange(sampling_feats,
                                          '(b n) c max_len d -> b n max_len (d c)',
                                          b=bs)

        #* ================== 5. 多相机特征回填与平均 ==================
        # 将各相机特征累加回网格，再除以有效相机数量，实现多视角平均。
        # 例如前相机与左前相机都有效：输出 (f_front + f_front_left) / 2。
        # 无有效相机的网格保持零；count 最小为 1，防止除零。
        feats_volume = x.new_zeros([bs, num_voxels, C])
        for j in range(bs):
            indexes_per_batch = indexes[j]
            for ii, index_query_per_img in enumerate(indexes_per_batch):
                feats_volume[j, index_query_per_img] += sampling_feats[j, ii, :len(index_query_per_img)]
        count = volume_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        feats_volume = feats_volume / count[..., None]

        #* ================== 6. 高度拼接与 BEV 输出 ==================
        # 形成 BEV：高度 z 与通道 c 拼接，不对高度求和。
        # [B, zs*xs*ys, C] -> [B, zs*C, ys, xs]；180*180*10、C=128 时为 [B,1280,180,180]。
        feats_volume = einops.rearrange(feats_volume,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.volume_size[2], h=self.volume_size[0], w=self.volume_size[1])
        if self.down_sample is not None:
            feats_volume = self.down_sample(feats_volume)
        return feats_volume

    @force_fp32()
    def point_sampling(self, camera2sensor, cam2imgs, post_rots, post_trans, bda):

        #* ================== 1. 撤销三维数据增强 ==================
        # 撤销 BEV/三维数据增强，恢复参考点的原始空间坐标。
        # 本函数仅做几何投影；points 参数中的 LiDAR 点不参与此处特征采样。
        B, N, _, _ = camera2sensor.shape
        num_points = self.ref_3d.shape[0]

        # [(NP 3) -> (1 NP 3)] - [(B 3) -> (B 1 3)] -> (B NP 3)
        reference_points = self.ref_3d.view(1, num_points, 3) - bda[:, :3, 3].view(B, 1, 3)
        # [(B 3 3) -> (B 1 3 3)] @ [(B NP 3) -> (B NP 3 1)] -> (B NP 3 1)
        reference_points = torch.inverse(bda[:, :3, :3].view(B, 1, 3, 3)).matmul(reference_points.unsqueeze(-1))
        # (B NP 3 1) -> (B 1 NP 3 1) -> (B 1 NP 3)
        reference_points = reference_points.view(B, 1, num_points, 3, 1).squeeze(-1)
        # (B 1 NP 3) - (B N 1 3) -> (B N NP 3) -> (B N NP 3 1)

        #* ================== 2. 变换到相机坐标并应用内参 ==================
        # p_cam = R^T (p_sensor - t)，再左乘内参 K。
        reference_points = (reference_points - camera2sensor[:, :, :3, 3].view(B, N, 1, 3)).unsqueeze(-1)
        # (B N 3 3)
        combine = cam2imgs.matmul(camera2sensor[:, :, :3, :3].transpose(-1, -2))
        # [(B N 3 3) -> (B N 1 3 3)] @ (B N NP 3 1) -> (B N NP 3 1) -> (B N NP 3)
        reference_points_img = combine.view(B, N, 1, 3, 3).matmul(reference_points).squeeze(-1)

        #* ================== 3. 透视除法 ==================
        # 保留相机前方的点，并除以相机深度得到像素坐标 (u, v)。
        eps = 1e-5
        # (B N NP 1)
        volume_mask = (reference_points_img[..., 2:3] > eps)
        # (B N NP 2)
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)

        #* ================== 4. 应用图像数据增强 ==================
        # 应用图像增强的旋转/缩放/平移，使坐标与当前图像特征对应。
        post_rots2 = post_rots[:, :, :2, :2]
        post_trans2 = post_trans[:, :, :2]
        # [(B N 2 2) -> (B N 1 2 2)] @ [(B N NP 2) -> (B N NP 2 1)] -> (B N NP 2 1)
        reference_points_img = post_rots2.view(B, N, 1, 2, 2).matmul(reference_points_img.unsqueeze(-1))
        # [(B N NP 2 1) -> (B N NP 2)] + [(B N 2) -> (B N 1 2)] -> (B N NP 2)
        reference_points_img = reference_points_img.squeeze(-1) + post_trans2.view(B, N, 1, 2)

        #* ================== 5. 归一化坐标与有效性筛选 ==================
        # 按输入图像尺寸归一化到 [0,1]，随后筛选图像边界内的点。
        #* 有效仅指深度为正且投影在图像内，不包含真实遮挡判断，也不提供深度概率权重。
        H_in, W_in = self.input_size
        reference_points_img[..., 0] /= W_in
        reference_points_img[..., 1] /= H_in

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                       & (reference_points_img[..., 1:2] < 1.0)
                       & (reference_points_img[..., 0:1] < 1.0)
                       & (reference_points_img[..., 0:1] > 0.0))

        volume_mask = torch.nan_to_num(volume_mask)

        # 原版保留所有相机的坐标和掩码，交给 forward 逐相机采样与平均。
        reference_points_img = reference_points_img.unsqueeze(-2).permute(1, 0, 2, 3, 4)  # num_cam, B, num_query, D, uv
        volume_mask = volume_mask.permute(1, 0, 2, 3)

        return reference_points_img, volume_mask


@VTRANSFORMS.register_module()
class BEVTransformV2(BaseModule):
    """
    Follow SparseBEV and optimize the bilinear interpolation sampling process.
    """
    def __init__(
            self,
            x: List[float], y: List[float], z: List[float],
            xs: int, ys: int, zs: int,
            input_size: List[int],
            in_channels: int = 256,
            out_channels: int = 128,
            top_type: str = 'lidar',
            down_sample: bool = False,
            down_sample_scale: int = 2,
            down_sample_channels: List[int] = [128*10, 64*10, 32*10, 16*10]
    ):
        """
        Args:
            x, y, z: Range in three directions.
            xs, ys, zs: Resolution in three directions.
            input_size (tuple(int)): Size of input images in format of (height, width).
            in_channels (int): Channels of input feature.
            out_channels (int): Channels of transformed feature.
            top_type (str): Coordinate system in which the task is labeled, 'lidar' or 'ego',
                        eg: 'lidar' for 3D detection and 'ego' for Occ3D-nuScenes.
        """
        super(BEVTransformV2, self).__init__()

        #* ================== 1. 创建三维参考网格 ==================
        # 初始化：在目标坐标系内建立固定的 3D 体素中心网格。
        # xs/ys/zs 为各轴网格数量；NP = xs * ys * zs，zs 是高度层数，不是图像深度 bin。
        self.pc_range = [x[0], y[0], z[0], x[1], y[1], z[1]]
        self.volume_size = [int(s) for s in [xs, ys, zs]]
        ref_3d = get_reference_points(self.pc_range[0], self.pc_range[3],
                                      self.pc_range[1], self.pc_range[4],
                                      self.pc_range[2], self.pc_range[5],
                                      self.volume_size[0], self.volume_size[1], self.volume_size[2])
        # ref_3d: [NP, 3]，随模型迁移设备，但不是可学习参数。
        self.register_buffer('ref_3d', ref_3d)
        # V2 仅在通道数不一致时做 1x1 映射，通道相同时直接使用输入特征。
        if in_channels != out_channels:
            self.transfer_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.transfer_conv = None
        self.out_channels = out_channels
        self.initial_flag = True
        self.input_size = input_size
        self.top_type = top_type

        #* ================== 2. 配置可选 BEV 卷积 ==================
        # 可选 BEV 卷积：输入通道应为 zs * out_channels。
        # scale=2 同时降低空间分辨率；scale=1 保持分辨率，只做卷积变换。
        if down_sample:
            assert down_sample_scale in [1, 2], 'only support 1x or 2x downsampling!'
            if down_sample_scale == 2:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True),
                    nn.Conv2d(
                        down_sample_channels[1],
                        down_sample_channels[2],
                        3,
                        stride=down_sample_scale,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(down_sample_channels[2]),
                    nn.ReLU(True),
                    nn.Conv2d(down_sample_channels[2], down_sample_channels[3], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[3]),
                    nn.ReLU(True),
                )
            else:  # down_sample_scale == 1
                assert len(down_sample_channels) == 2, "the len of 'down_sample_channels: List[int]' should be 2!"
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True)
                )
        else:
            self.down_sample = None
        self.fp16_enabled = False  # TODO: important!!! # will be automatically set as true

    @force_fp32()
    def forward(self,
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs,
                ):
        """Transform image-view feature into bird-eye-view feature.

        Args:
        Returns:
        """

        #* ================== 1. 准备包含相机轴的五维特征 ==================
        # 准备 V2 的五维采样输入：[B,N,C,H,W] -> [B,C,N,H,W]。
        # 将 N 作为 grid_sample 的第三个空间轴；它表示相机编号，不是物理深度。
        B = x.shape[0]
        if self.transfer_conv is not None:
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        #* ================== 2. 确定目标坐标系 ==================
        # 确定投影的目标坐标系，camera2sensor 表示相机到该坐标系的变换。
        # ego：经 global 对齐到第一个相机对应的关键帧 ego；lidar：直接使用 camera2lidar。
        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)  # (B, 1, 4, 4)
            global2keyego = torch.inverse(keyego2global.double())  # (B, 1, 4, 4)
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()  # (B, N_views, 4, 4)
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError
        # 将固定 3D 网格投影到图像；这里只计算采样坐标和有效掩码，还未读取图像特征。
        reference_points_img, volume_mask = self.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
                                                                img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                                lidar_aug_matrix)

        #* ================== 3. 按 batch 整理有效采样点 ==================
        # 每个网格点已选好一个相机；按 batch 收集有效点，补齐后统一采样。
        # reference_points_img: [B,NP,3]，最后一维为 (u,v,归一化相机编号)。
        # 下方固定数字只是某个配置的示例，实际 NP 和有效点数由配置、相机姿态决定。
        bs, num_points, _ = reference_points_img.shape  # num_points: 324000
        valid_index_per_batch = [volume_mask[i].squeeze(-1).nonzero().squeeze(-1) for i in range(bs)]  # [torch.Size([312387])]
        max_len = max([len(per_batch) for per_batch in valid_index_per_batch])

        reference_points_rebatch = reference_points_img.new_zeros([bs, max_len, 3])
        for j in range(bs):
            reference_points_rebatch[j, :len(valid_index_per_batch[j])] = reference_points_img[j, valid_index_per_batch[j]]
        reference_points_rebatch = reference_points_rebatch.view(bs, max_len, 1, 1, 3)
        reference_points_rebatch = 2 * reference_points_rebatch - 1  # torch.Size([1, 312387, 1, 1, 3])

        #* ================== 4. 带相机坐标的特征采样 ==================
        # V2 每个有效网格点只采样一次，避免同一点在重叠视野中重复读取多个相机的特征。
        #* 五维输入配合 bilinear 模式实际执行三线性插值。
        # 相机坐标精确落在选定相机切片上（align_corners=True），因此不主动混合不同相机。
        # 图像平面内仍有插值权重，但没有沿物理深度的概率加权。
        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c max_len 1 1 -> b max_len c',
                                          b=bs)

        #* ================== 5. 单相机特征直接回填 ==================
        # 直接把选定相机的采样结果写回网格；V2 不再累加和平均多个相机。
        # 例如前相机与左前相机都有效且前相机排在前面：只输出 f_front，不融合另一个视角。
        # 相机顺序会影响重叠区域的选择；此处省去了原版的逐相机累加、计数与除法。
        # 补齐位置不写回，无有效相机的网格保持零。
        feats_volume = x.new_zeros([bs, num_points, self.out_channels])
        for j in range(bs):
            feats_volume[j, valid_index_per_batch[j]] = sampling_feats[j, :len(valid_index_per_batch[j])]

        #* ================== 6. 高度拼接与 BEV 输出 ==================
        # 形成 BEV：高度 z 与通道 c 拼接，不对高度求和。
        # [B, zs*xs*ys, C] -> [B, zs*C, ys, xs]；180*180*10、C=128 时为 [B,1280,180,180]。
        feats_volume = einops.rearrange(feats_volume,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.volume_size[2], h=self.volume_size[0], w=self.volume_size[1])
        if self.down_sample is not None:
            feats_volume = self.down_sample(feats_volume)
        return feats_volume

    @force_fp32()
    def point_sampling(self, camera2sensor, cam2imgs, post_rots, post_trans, bda, mode='fix'):

        #* ================== 1. 撤销三维数据增强 ==================
        # 撤销 BEV/三维数据增强，恢复参考点的原始空间坐标。
        # 本函数仅做几何投影；points 参数中的 LiDAR 点不参与此处特征采样。
        B, N, _, _ = camera2sensor.shape
        num_points = self.ref_3d.shape[0]

        # [(NP 3) -> (1 NP 3)] - [(B 3) -> (B 1 3)] -> (B NP 3)
        reference_points = self.ref_3d.view(1, num_points, 3) - bda[:, :3, 3].view(B, 1, 3)
        # [(B 3 3) -> (B 1 3 3)] @ [(B NP 3) -> (B NP 3 1)] -> (B NP 3 1)
        reference_points = torch.inverse(bda[:, :3, :3].view(B, 1, 3, 3)).matmul(reference_points.unsqueeze(-1))
        # (B NP 3 1) -> (B 1 NP 3 1) -> (B 1 NP 3)
        reference_points = reference_points.view(B, 1, num_points, 3, 1).squeeze(-1)
        # (B 1 NP 3) - (B N 1 3) -> (B N NP 3) -> (B N NP 3 1)

        #* ================== 2. 变换到相机坐标并应用内参 ==================
        # p_cam = R^T (p_sensor - t)，再左乘内参 K。
        reference_points = (reference_points - camera2sensor[:, :, :3, 3].view(B, N, 1, 3)).unsqueeze(-1)
        # (B N 3 3)
        combine = cam2imgs.matmul(camera2sensor[:, :, :3, :3].transpose(-1, -2))
        # [(B N 3 3) -> (B N 1 3 3)] @ (B N NP 3 1) -> (B N NP 3 1) -> (B N NP 3)
        reference_points_img = combine.view(B, N, 1, 3, 3).matmul(reference_points).squeeze(-1)

        #* ================== 3. 透视除法 ==================
        # 保留相机前方的点，并除以相机深度得到像素坐标 (u, v)。
        eps = 1e-5
        # (B N NP 1)
        volume_mask = (reference_points_img[..., 2:3] > eps)
        # (B N NP 2)
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)

        #* ================== 4. 应用图像数据增强 ==================
        # 应用图像增强的旋转/缩放/平移，使坐标与当前图像特征对应。
        post_rots2 = post_rots[:, :, :2, :2]
        post_trans2 = post_trans[:, :, :2]
        # [(B N 2 2) -> (B N 1 2 2)] @ [(B N NP 2) -> (B N NP 2 1)] -> (B N NP 2 1)
        reference_points_img = post_rots2.view(B, N, 1, 2, 2).matmul(reference_points_img.unsqueeze(-1))
        # [(B N NP 2 1) -> (B N NP 2)] + [(B N 2) -> (B N 1 2)] -> (B N NP 2)
        reference_points_img = reference_points_img.squeeze(-1) + post_trans2.view(B, N, 1, 2)

        #* ================== 5. 归一化坐标与有效性筛选 ==================
        # 按输入图像尺寸归一化到 [0,1]，随后筛选图像边界内的点。
        #* 有效仅指深度为正且投影在图像内，不包含真实遮挡判断，也不提供深度概率权重。
        H_in, W_in = self.input_size
        reference_points_img[..., 0] /= W_in  # 使用配置中的图像宽度
        reference_points_img[..., 1] /= H_in  # 使用配置中的图像高度

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                       & (reference_points_img[..., 1:2] < 1.0)
                       & (reference_points_img[..., 0:1] < 1.0)
                       & (reference_points_img[..., 0:1] > 0.0))

        volume_mask = torch.nan_to_num(volume_mask)

        # (B N NP 2) -> (B NP N 2)
        reference_points_img = reference_points_img.permute(0, 2, 1, 3)
        # (B N NP 1) -> (B NP N 1)
        volume_mask = volume_mask.permute(0, 2, 1, 3)

        #* ================== 6. V2 选择单个有效相机 ==================
        # 对布尔有效掩码取 argmax，选编号最小的有效相机。
        # 这不是按置信度或距离选相机；全无效时虽返回编号 0，但掩码仍为 False，之后不会采样写回。
        if mode == 'fix':
            idx_camera = torch.argmax(volume_mask.squeeze(-1).float(), dim=-1)
        elif mode == 'random':
            raise NotImplementedError
        else:
            raise NotImplementedError

        idx_batch = torch.arange(B, dtype=torch.long, device=reference_points_img.device)
        idx_point = torch.arange(num_points, dtype=torch.long, device=reference_points_img.device)
        idx_batch = idx_batch.view(B, 1).expand(B, num_points)
        idx_point = idx_point.view(1, num_points).expand(B, num_points)
        reference_points_img = reference_points_img[idx_batch, idx_point, idx_camera]
        volume_mask = volume_mask[idx_batch, idx_point, idx_camera]

        #* ================== 7. V2 编码相机采样坐标 ==================
        # 将相机编号映射到 [0,1]，拼成 (u,v,camera)。
        # forward 再映射到 [-1,1]；配合 align_corners=True 可对应到准确相机切片。
        # 此编号归一化写法要求 N > 1。
        coors_camera = idx_camera[..., None].float() / (N - 1)
        reference_points_img = torch.cat([reference_points_img, coors_camera], dim=-1)

        return reference_points_img, volume_mask
