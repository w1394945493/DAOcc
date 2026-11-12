from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["VisualDualBranchFuser"]


@FUSERS.register_module()
class VisualDualBranchFuser(nn.Module):
    """仅使用相机 BEV、但保持原双分支拼接结构的融合模块。"""

    def __init__(
        self,
        camera_channels: int,
        auxiliary_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.camera_channels = camera_channels
        self.auxiliary_channels = auxiliary_channels
        self.out_channels = out_channels

        # 相机 BEV 为 [B,1280,180,180]。用一个 3×3 卷积生成 256 通道的
        # 视觉辅助分支，以替代原 LiDAR SparseEncoder 输出的 256 通道 BEV。
        # 该分支约 295 万参数，与被移除的 LiDAR backbone 约 269 万参数接近。
        self.visual_auxiliary = nn.Sequential(
            nn.Conv2d(
                camera_channels,
                auxiliary_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(auxiliary_channels),
            nn.ReLU(inplace=True),
        )

        # 保持原 ConvFuser 的结构与通道数：1280 通道原始相机 BEV 与
        # 256 通道视觉辅助 BEV 拼接为 1536 通道，再融合为 512 通道。
        self.fusion = nn.Sequential(
            nn.Conv2d(
                camera_channels + auxiliary_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # camera-only 配置关闭了 LiDAR encoder，因此这里只应收到一个相机 BEV。
        # 显式检查可以在配置错误时立即报出清楚的原因，避免后续卷积通道报错。
        if len(inputs) != 1:
            raise ValueError(
                "VisualDualBranchFuser 只接受一个相机 BEV，"
                f"实际收到 {len(inputs)} 个输入"
            )

        camera_bev = inputs[0]
        if camera_bev.shape[1] != self.camera_channels:
            raise ValueError(
                "相机 BEV 通道数与配置不一致："
                f"期望 {self.camera_channels}，实际 {camera_bev.shape[1]}"
            )

        visual_auxiliary = self.visual_auxiliary(camera_bev)
        return self.fusion(torch.cat([camera_bev, visual_auxiliary], dim=1))
