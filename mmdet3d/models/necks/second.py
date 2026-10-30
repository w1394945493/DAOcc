# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS

from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks import ConvModule

__all__ = ["SECONDFPN", "FPN_LSS", "LSSFPN3D"]


@NECKS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                )
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                )

            deblock = nn.Sequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type="Kaiming", layer="ConvTranspose2d"),
                dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


@NECKS.register_module()
class FPN_LSS(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super(FPN_LSS, self).__init__()
        # 当前DAOcc：选第0和第2层，scale_factor=2、extra_upsample=None。
        # 此模块融合不同深度的BEV特征；名字中的LSS不表示这里执行图像视角变换。
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        # 用于上采样high-level的feature map
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        # --------------------------------------------------------------------------------------#
        # 当前配置的两层3×3卷积：640→512→512，每层后接BN、ReLU。
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        """
        Args:
            feats: List[Tensor,] multi-level features
                List[(B, C1, H, W), (B, C2, H/2, W/2), (B, C3, H/4, W/4)]
        Returns:
            x: (B, C_out, 2*H, 2*W)
        """
        #* =================== 1. 取浅层与深层BEV特征 ===================
        # 上方docstring是通用示意；以下尺寸按当前DAOcc配置说明。
        # 默认索引(0,2)：x2=[B,128,180,180]，x1=[B,512,90,90]。
        # 中间第1层不直接拼接，但它参与了生成第2层特征的计算。
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        # 可选的浅层通道映射；当前配置lateral=None，不执行。
        if self.lateral:
            x2 = self.lateral_conv(x2)
        #* =================== 2. 对齐空间尺寸并拼接 ===================
        # 深层双线性上采样90×90→180×180，再与浅层沿通道拼接：128+512=640。
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        #* =================== 3. 两层卷积整合特征 ===================
        # [B,640,180,180] → [B,512,180,180]，融合浅层空间细节与深层信息。
        x = self.conv(x1)   # (B, C', H, W)
        #* =================== 4. 可选额外上采样并返回 ===================
        # 当前extra_upsample=null，此分支不执行，最终保持180×180。
        # 返回的是检测头与占据头共享的BEV，不是最终三维占据标签。
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x


@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU', inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        """
        Args:
            feats: List[
                (B, C, Dz, Dy, Dx),
                (B, 2C, Dz/2, Dy/2, Dx/2),
                (B, 4C, Dz/4, Dy/4, Dx/4)
            ]
        Returns:
            x: (B, C, Dz, Dy, Dx)
        """
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)       # (B, 2C, Dz, Dy, Dx)
        x_32 = self.up2(x_32)       # (B, 4C, Dz, Dy, Dx)
        x = torch.cat([x_8, x_16, x_32], dim=1)     # (B, 7C, Dz, Dy, Dx)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)    # (B, C, Dz, Dy, Dx)
        return x
