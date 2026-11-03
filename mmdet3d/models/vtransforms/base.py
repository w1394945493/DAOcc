from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]


#* ==================== 网格参数：步长、首格中心、格子数量 ====================
# 各bound为[起点,终点,步长]；dx/bx/nx均为[3]，依次对应物理X/Y/Z轴。
# dx为体素边长，bx为第一个体素中心，nx为各轴体素数；范围应能被步长整除。
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


#* 普通LSS基类：构建视锥 → 三维定位 → 子类生成深度加权特征 → BEV池化。
# B=batch，N=相机数，D=相机深度bin数，fH/fW=图像特征尺寸，Z=高度格子数。
# D沿相机光轴，Z沿LiDAR高度轴，二者不是同一个维度。
class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        # 固定网格参数随模型迁移设备/保存权重，但requires_grad=False，不参与学习。
        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # self.C是每个三维采样点的内容特征通道数；最终BEV拼接高度后为Z*C通道。
        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        # 由混合精度包装器启用；force_fp32在启用时将半精度浮点输入转回FP32。
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        #* ==================== 1. 构建视锥采样网格 ====================
        # 这里生成的是(u,v,depth)，不是LiDAR坐标；每个特征像素沿深度取D个样本。
        iH, iW = self.image_size
        fH, fW = self.feature_size

        # [D,fH,fW]，arange不包含dbound终点；深度为相机Z坐标，单位米。
        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        # xs/ys：[D,fH,fW]，使用输入图像像素坐标而非特征图索引。
        # 横向覆盖0..iW-1，纵向覆盖0..iH-1。
        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # [D,fH,fW,3]；后续通过广播共享给每个batch和相机。
        frustum = torch.stack((xs, ys, ds), -1)  # torch.Size([118, 32, 88, 3])
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        #* ==================== 2. 视锥点转换到增强后的LiDAR坐标 ====================
        # camera2lidar_rots/intrins/post_rots：[B,N,3,3]；对应平移：[B,N,3]。
        # 顺序：撤销图像增强 → 像素反投影 → 相机到LiDAR → 应用三维增强。
        B, N, _ = camera2lidar_trans.shape

        # 2.1 撤销图像增强的平移和旋转/缩放，恢复原始图像上的采样坐标。
        # 减平移后[B,N,D,fH,fW,3]；矩阵乘法后多一个末尾单例维度。
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # 2.2 (u,v,d)转换为(u*d,v*d,d)，再乘K逆矩阵还原相机三维坐标。
        # 合并R_camera2lidar与K逆矩阵，一次乘法后加平移得到LiDAR XYZ。
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # 2.3 应用与点云一致的三维增强；extra_rots：[B,3,3]，extra_trans：[B,3]。
        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        # [B,N,D,fH,fW,3]，每个视锥采样点在增强后的LiDAR坐标系中的物理位置。
        return points
    #! -----------------------------------------------------#
    #! 子类负责具体的概率深度预测
    # 子类实现内容特征×深度概率；输出[B,N,D,fH,fW,C]。基类不预测深度。
    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        #* ==================== 3. BEV pooling：按三维格子累加特征 ====================
        # geom_feats：[B,N,D,H,W,3]物理坐标；x：[B,N,D,H,W,C]深度加权特征。
        # 此处H/W是图像特征尺寸，不是最终BEV尺寸。
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # 3.1 展平全部batch、相机、深度和像素位置，得到[Nprime,C]。
        x = x.reshape(Nprime, C)

        # 3.2 (物理坐标-网格起点)/步长，转换为整数体素索引，再补上batch编号。
        # 注意.long()向零截断；负的小数与floor不同，此处仅解释原逻辑、不修改。
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # x, y, z, bs

        # 3.3 按整数索引过滤网格外的样本；保留后x为[M,C]，索引为[M,4]。
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # 3.4 CUDA算子将相同(batch,x,y,z)的特征求和，不按点数/相机数取平均。
        # 输出[B,C,Z,X,Y]；多个相机和深度样本可以落入同一体素，空格子为0。
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # 3.5 沿高度拆成Z个[B,C,X,Y]，再拼接通道，得到[B,Z*C,X,Y]。
        # 不是对Z求均值/求和；通道排列为z0的C通道、z1的C通道……。
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        **kwargs,
    ):
        #* ==================== 4. 普通LSS完整前向 ====================
        # img为[B,N,Cin,fH,fW]图像特征；points虽在接口中，但本路径不使用点云深度。
        # 下面rots/trans及lidar2ego分解结果当前未参与运算；实际使用camera2lidar。
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # 子类LSSTransform根据图像预测深度概率；先提升到视锥，再池化为BEV。
        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


#* LiDAR引导LSS基类：比普通LSS多出“将点云投影成稀疏深度图”步骤。
# 深度图作为子类网络输入，不直接替代预测的深度概率，也不构成深度监督损失。
class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        #* ==================== 5. 提取相机标定与数据增强参数 ====================
        # img：[B,N,Cin,fH,fW]；points：长度B的列表，每项[Pi,点特征维数]。
        # 各相机变换通常为[B,N,4,4]，LiDAR增强矩阵为[B,4,4]。
        # sensor2ego/lidar2ego的局部分解保留原实现，实际投影依赖lidar2image。
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        #* ==================== 6. 逐帧构造LiDAR稀疏深度图 ====================
        # depth：[B,N,1,Hi,Wi]，分辨率是输入图像而非图像特征；未投影到点处为0。

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )

        for b in range(batch_size):
            # XYZ切片共享原点云存储；后面的-=会原地修改输入点云！
            # DepthLSSTransformDAOcc已在调用父类前clone点云，原类行为保持不变。
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # 6.1 撤销LiDAR三维增强：[Pi,3]减平移，再乘逆矩阵，得到[3,Pi]。
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # 6.2 用lidar2image同时投影到N个相机，得到[N,3,Pi]齐次图像坐标。
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # 6.3 第三维是相机光轴深度；前两维除以深度，得到像素(u,v)。
            # dist为[N,Pi]视图；下面深度clamp也会影响它，并非独立保存原始深度。
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # 6.4 应用图像增强，与实际输入图像对齐，取像素坐标[N,Pi,2]。
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # 6.5 将(u,v)交换为(行v,列u)，供深度图索引；这里并未归一化到[-1,1]。
            cur_coords = cur_coords[..., [1, 0]]

            # 仅检查图像边界；原实现此处没有单独的正深度或遮挡筛选。
            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            # 6.6 逐相机填入深度值；long转换到像素索引。
            # 多个点落在同一像素时采用原来的重复索引赋值，不保证保留最近点，
            # 也没有求均值或深度补全；因此得到的是稀疏辅助输入。
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        #* ==================== 7. 深度引导预测与BEV聚合 ====================
        # geom已将视锥网格映射到增强后的LiDAR系：[B,N,D,fH,fW,3]。
        # 子类编码depth并拼接图像，预测概率后得到[B,N,D,fH,fW,C]特征。
        # 池化并拼接高度后返回[B,Z*C,X,Y]；DAOcc适配类再交换X/Y。
        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x
