from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
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
        downsample: int = 1,
        dt_strides: Tuple[int, int] = [4, 2],
    ) -> None:
        #* ==================== 1. 初始化三维网格与相机深度采样 ====================
        # image_size=(Hi,Wi)是输入图像尺寸；feature_size=(fH,fW)是图像neck输出尺寸。
        # x/y/zbound=[起点,终点,步长]定义LiDAR坐标系的体素网格。
        # dbound定义相机光轴Z方向的深度采样，区别于体素网格的高度zbound。
        # 父类设置self.C=out_channels、self.D=深度bin数，并构建视锥采样网格。
        # 例如dbound=[1,60,0.5]时，采样1.0至59.5米，共118个深度位置。
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        #* ==================== 2. LiDAR稀疏深度图编码 ====================
        # 父类将真实点云投影到各相机，生成[B,N,1,Hi,Wi]深度图，无点位置为0。
        # 合并B和相机数N后，经此网络得到[B*N,64,fH,fW]深度特征。
        # 默认步长4×2=8；输出空间尺寸必须与图像特征匹配，才能执行通道拼接。
        # 这是辅助输入编码，不是把LiDAR深度直接转换成one-hot深度概率。
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=dt_strides[0], padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=dt_strides[1], padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        #* ==================== 3. 联合预测深度分布与内容特征 ====================
        # 输入为图像Cin通道加深度编码64通道；输出为D+C通道。
        # 前D通道是深度logits，后C通道是待提升到三维空间的内容特征。
        # 网络需要训练才能学会利用LiDAR深度；本类没有单独的深度监督损失。
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        #* ==================== 4. 可选BEV下采样 ====================
        # BEV pooling后将高度Z拼入通道，输入实际为[B,Z*C,X,Y]。
        # 下方卷积仅按C通道构建，因此多高度层Z>1时不能直接启用此下采样。
        # DAOcc适配版本采用downsample=1，保留空间尺寸和全部高度通道。
        if downsample > 1:
            assert downsample == 2, downsample # 如果配置downsample：2：用卷积将空间尺寸缩小为原来一半（长宽各缩小一半）
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    #* ----------------------------------------------------------------------#
    #* 概率深度预测
    # 混合精度启用且fp16_enabled=True时，装饰器将浮点输入转回FP32计算。
    @force_fp32()
    def get_cam_feats(self, x, d):
        #* ==================== 5. 合并batch与相机维度 ====================
        # x：[B,N,Cin,fH,fW]图像特征；d：[B,N,1,Hi,Wi]稀疏深度图。
        # 此处局部变量C表示输入Cin；self.C才是输出内容特征通道数。
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        #* ==================== 6. 深度特征与图像特征拼接 ====================
        d = self.dtransform(d)  # [B*N,64,fH,fW]：编码后的深度特征
        x = torch.cat([d, x], dim=1)  # [B*N,64+Cin,fH,fW]
        x = self.depthnet(x)  # [B*N,D+self.C,fH,fW]

        #* ==================== 7. 深度概率加权，提升为视锥三维特征 ====================
        # depth：[B*N,D,fH,fW]；每个特征像素沿D维的概率之和为1。
        # 不取argmax或期望深度：保留全部深度，每个位置的内容特征按概率分配。
        depth = x[:, : self.D].softmax(dim=1)
        # [B*N,1,D,fH,fW] × [B*N,self.C,1,fH,fW]
        # -> [B*N,self.C,D,fH,fW]；同一像素在不同深度共享内容向量但权重不同。
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        #* ==================== 8. 恢复多相机布局供BEV pooling使用 ====================
        x = x.view(B, N, self.C, self.D, fH, fW)  # [B,N,Cout,D,fH,fW]
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B,N,D,fH,fW,Cout]
        # 返回的仍是视锥特征，还不是BEV；三维定位和池化由父类forward完成。
        return x

    def forward(self, *args, **kwargs):
        #* ==================== 9. 完整视角变换与BEV输出 ====================
        # BaseDepthTransform.forward依次完成：
        # ① 点云投影生成稀疏深度图；② 用内外参和增强矩阵定位视锥三维采样点；
        # ③ 调用上面的get_cam_feats预测深度并加权；④ 将落入同一体素的特征累加；
        # ⑤ 将Z层沿通道拼接，得到[B,Z*self.C,X,Y]，不是对高度取平均。
        # 本原始实现不交换X/Y；DepthLSSTransformDAOcc另外适配为[B,Z*C,Y,X]。
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x
