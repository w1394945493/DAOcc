from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 当前配置：相机1280通道 + LiDAR256通道 = 1536，融合后输出512通道。
        # 3×3卷积同时混合模态通道及邻域信息；padding=1保持BEV空间尺寸。
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        #* =================== 1. 沿通道维拼接两种模态 ===================
        # inputs内的特征已位于共同BEV网格，要求batch及空间尺寸一致。
        # 当前配置：[B,1280,180,180]与[B,256,180,180]拼为[B,1536,180,180]。
        #* =================== 2. 用卷积学习融合特征 ===================
        # super().forward调用上面nn.Sequential中的Conv2d → BN → ReLU。
        # 返回[B,512,180,180]；这里还不预测检测框或占据类别。
        return super().forward(torch.cat(inputs, dim=1))
