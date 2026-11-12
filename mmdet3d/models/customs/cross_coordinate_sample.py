from typing import List
import warnings
import torch
import einops
import torch.nn.functional as F
from torch import nn
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

from mmdet3d.models.builder import CUSTOMS
from ..utils.reference_poinst import get_reference_points

__all__ = ["CrossCoordinateSample"]


@CUSTOMS.register_module()
class CrossCoordinateSample(BaseModule):
    # TODO: [yz] make CrossCoordinateSample more generalized
    def __init__(self, point_range: list, point_num: list, lidar_point_range: list, point_type: str = 'ego',
                 extra_up: bool = False, extra_up_scale: int = 2, in_dim: int = 512, out_dim: int = 128,
                 norm_cfg=dict(type='BN')) -> None:
        """Convert points in ego/lidar coordinate to lidar coordinate and sample features corresponding to
        the converted points.
        Args:
            point_range (list[float]): [h_min, h_max, w_min, w_max, z_min, z_max], h<->y, w<->x.
            point_num (list[int]): [h_num, w_num, z_num], h<->y, w<->x.
            point_type (str): ego (Occ3D) or lidar (SurroundOcc and OpenOccupancy).
            lidar_point_range (list[float]): [y_min, y_max, x_min, x_max]
        """
        super().__init__()
        #! BVRE 核心 1：point_range 定义目标 OCC 的 XY 范围，point_num 定义采样网格数量。
        #! 当前 Occ3D 配置为自车系 XY 各 [-40,40] m、200×200 格，即平面分辨率 0.4 m。
        # point_range 顺序为 [ymin,ymax,xmin,xmax,zmin,zmax]；point_num 为 [Y,X,Z]。
        # get_reference_points 在每格中心生成 (x,y,z)，此配置 XY 中心从 -39.8 到 39.8 m。
        #! zmin=zmax=0、Z=1 仅表示查询 z=0 平面的 BEV 特征，不定义最终 OCC 的高度范围。
        # 最终高度层数由 BEVOCCHead2D.Dz=16 决定，各层物理高度由占据真值网格定义。
        ref_points = get_reference_points(*point_range, *point_num)
        self.register_buffer('ref_points', ref_points)
        #! lidar_point_range 定义源 BEV 的覆盖范围；当前为 LiDAR 系 XY 各 [-54,54] m。
        # BVRE 的范围扩展在上游点云/相机编码配置中完成；本模块将大范围特征对齐到目标 OCC 网格。
        # 源范围用于归一化采样坐标，不是最终 OCC 输出范围；源/目标还可能处于不同坐标系。
        if len(lidar_point_range) == 4:
            warnings.warn("4-element lidar_point_range is deprecated, please use 6-element format")
            self.lidar_y_min, self.lidar_y_max, self.lidar_x_min, self.lidar_x_max = lidar_point_range
        else:
            assert len(lidar_point_range) == 6
            self.lidar_x_min, self.lidar_y_min, _, self.lidar_x_max, self.lidar_y_max, _ = lidar_point_range
        # 输出行数为 Y 格数、列数为 X 格数；当前为 200×200。
        # 下方 reshape 只恢复二维平面，本实现的使用配置要求 Z 采样数为 1。
        self.output_h, self.output_w = point_num[:2]
        assert point_type in ['ego', 'lidar']
        self.point_type = point_type
        if in_dim != out_dim:
            self.transfer_conv = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.transfer_conv = None
        self.extra_up = extra_up
        if extra_up:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=extra_up_scale, mode='bilinear', align_corners=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_dim)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)
            )
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, x: torch.Tensor,
                lidar_aug_matrix: torch.Tensor, lidar2ego: torch.Tensor, occ_aug_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features in lidar coordinate with shape (N, C, H, W).
            lidar_aug_matrix (torch.Tensor): (N, 4, 4)
            lidar2ego (torch.Tensor): (N, 4, 4)
            occ_aug_matrix (torch.Tensor): (N, 4, 4)
        Returns:
            x (torch.Tensor):
        """
        #! 论文 BVRE 公式：F_occ = GridSample(F_r, Norm(T_o2l @ P_o))。
        # 整体功能：从已有的Lidar BEV特征图中，取出“目标占据网格每个位置应该使用的特征”
        # =================== 1. 调整通道，准备目标网格 ===================
        # 输入是 LiDAR 系 BEV [B,C,H,W]；当前配置先用 1×1 卷积将 512 通道变为 128。
        # 这一步只调整通道，不改变空间位置，也不是坐标变换。
        if self.transfer_conv is not None:
            x = self.transfer_conv(x)
        batch_size = x.shape[0]
        B = lidar_aug_matrix.shape[0]
        num_points = self.ref_points.shape[0]
        # ref_points 是初始化时按目标范围生成的查询点，不是输入点云。
        # 两套配置都查询 200×200 个平面位置（z=0），不是直接在这里生成16层占据。
        # =================== 2. 将目标查询点表示到输入 BEV 的坐标系 ===================
        if self.point_type == 'ego':
            # Occ3D：目标点属于自车系，必须映射到 LiDAR BEV 上才能取特征。
            #! 先撤销 OCC 增强，保证目标查询点可用原始 lidar2ego 外参变换。
            # 本实现只使用 occ_aug_matrix 左上3×3。
            # 普通无增强推理时增强矩阵为单位阵。
            # inverse occ data augment
            # ([B 3 3] -> [B 1 3 3]) @ ([num_points 3] -> [1 num_points 3 1]) -> [B num_points 3 1]
            ref_ego = torch.inverse(occ_aug_matrix[:, :3, :3].view(B, 1, 3, 3)).matmul(self.ref_points.view(1, num_points, 3, 1))
            # [B num_points 3 1] -> [B num_points 3 1] -> [B num_points 3]
            ref_ego = ref_ego.view(B, num_points, 3, 1).squeeze(-1)

            #! BVRE 核心 2：实现 T_o2l @ P_o，自车系 → LiDAR 系：p_lidar = inverse(R) @ (p_ego - t)。
            # lidar2ego 表示正向的 LiDAR → 自车变换，此处因此使用逆变换。
            # 注意：变换的是采样位置，不是逐个移动输入 BEV 的像素。
            # ego to lidar
            # [B num_points 3] - ([B 3] -> [B 1 3]) -> [B num_points 3] -> [B num_points 3 1]
            ref_lidar = (ref_ego - lidar2ego[:, :3, 3].view(batch_size, 1, 3)).unsqueeze(-1)
            # ([B 3 3] -> [B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1] -> [B num_points 3]
            # ref_lidar = lidar2ego[:, :3, :3].transpose(-1, -2).view(batch_size, 1, 3, 3).matmul(ref_lidar).squeeze(-1)
            # ([B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1]
            ref_lidar = torch.inverse(lidar2ego[:, :3, :3]).view(batch_size, 1, 3, 3).matmul(ref_lidar)

            #! 将查询位置施加 LiDAR 增强，使采样坐标与增强后的源 BEV 对齐。
            # lidar data augment
            # ([B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1]
            ref_lidar = lidar_aug_matrix[:, :3, :3].view(B, 1, 3, 3).matmul(ref_lidar.view(B, num_points, 3, 1))
            # ([B num_points 3 1] -> [B num_points 3]) + ([B 3] -> [B 1 3]) -> [B num_points 3]
            ref_lidar = ref_lidar.squeeze(-1) + lidar_aug_matrix[:, :3, 3].unsqueeze(1)
            # [B num_points 3] -> [B num_points 1 3] -> [B num_points 1 2]
            ref_lidar = ref_lidar.unsqueeze(-2)[:, :, :, :2]
        else:  # self.point_type == 'lidar'
            # SurroundOcc：查询点本来就在 LiDAR 系，直接取 XY，无需 lidar2ego 逆变换。
            # 此分支实际没有应用增强矩阵；当前 SurroundOcc 配置关闭了3D几何增强。
            # 仍需将目标 ±50 m、200×200 网格采样到输入 ±54 m 的 BEV 上。
            ref_lidar = self.ref_points[:, :2].unsqueeze(-2).unsqueeze(0).repeat(B, 1, 1, 1)

        #! BVRE 核心 3：Norm，将 LiDAR 米制坐标按源 BEV 范围映射到 grid_sample 坐标。
        # 源范围内的点先映射到 [0,1]，再映射到 [-1,1]；越界点不会被裁剪。
        # 当前 x_norm = 2*(x_lidar+54)/108-1，Y 同理；这里必须使用源范围 ±54，而非目标 ±40。
        # 最后一维的顺序是 (x,y)，分别对应特征图的列、行。
        lidar_x_length, lidar_y_length = self.lidar_x_max-self.lidar_x_min, self.lidar_y_max-self.lidar_y_min
        ref_lidar[..., 0] = (ref_lidar[..., 0] - self.lidar_x_min) / lidar_x_length
        ref_lidar[..., 1] = (ref_lidar[..., 1] - self.lidar_y_min) / lidar_y_length
        ref_lidar = ref_lidar * 2 - 1

        #! BVRE 核心 4：GridSample，从大范围融合 BEV 插值获得目标 OCC 网格的特征。
        # 对每个目标位置，从原 BEV 邻近像素插值得到一个特征向量。
        # 查询网格 [B,N,1,2] → 采样结果 [B,C,N,1]；越界位置补零。
        # align_corners=False 按像素边界解释归一化范围。
        x = F.grid_sample(
            x,
            ref_lidar,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        #! 将采样结果恢复为目标 OCC 平面：[B,128,200,200]，其空间范围由目标 point_range 决定。
        # 这是按物理坐标重采样，不是直接裁剪数组；最终 16 层语义预测由后续占据头完成。
        x = einops.rearrange(x.squeeze(-1), 'bs c (h w) -> bs c h w', h=self.output_h, w=self.output_w)

        # =================== 5. 可选上采样并返回 ===================
        # 当前两套配置未启用 extra_up；返回特征供占据头继续分类。
        # 输出对应的坐标系由目标网格决定：Occ3D 为自车系，SurroundOcc 为 LiDAR 系。
        if self.extra_up:
            x = self.up(x)

        return x
