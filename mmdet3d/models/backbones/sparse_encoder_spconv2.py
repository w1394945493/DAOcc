# Copyright (c) OpenMMLab. All rights reserved.
"""使用官方 spconv 2.x 实现的独立稀疏编码器。

本文件不修改仓库原有的 ``SparseEncoder``。配置中将 backbone 的
``type`` 设为 ``SparseEncoderSpconv2`` 时才会使用这里的实现。
"""

from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16
from torch import nn

import spconv.pytorch as spconv
from mmdet.models import BACKBONES


def _reverse_3d(value):
    """把仓库旧版 [X, Y, Z] 参数转换为 spconv2 的 [Z, Y, X]。"""
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"三维稀疏卷积参数必须包含 3 项，实际为: {value}")
        return tuple(reversed(value))
    return value


def _replace_feature(x, features):
    """统一通过 spconv2 接口替换特征，避免直接修改只读属性。"""
    return x.replace_feature(features)


def make_sparse_convmodule_spconv2(
    in_channels,
    out_channels,
    kernel_size,
    indice_key,
    stride=1,
    padding=0,
    conv_type="SubMConv3d",
    norm_cfg=None,
    order=("conv", "norm", "act"),
):
    """构建完全由 spconv2 组成的稀疏卷积模块。

    参数仍采用 DAOcc 旧实现的 [X, Y, Z] 顺序，并在此处统一转换，
    因而原 SparseEncoder 的配置可以原样复用。
    """
    if not isinstance(order, tuple) or len(order) > 3:
        raise TypeError("order 必须是长度不超过 3 的 tuple")
    if not set(order).issubset({"conv", "norm", "act"}):
        raise ValueError(f"order 包含不支持的操作: {order}")

    conv_classes = {
        "SubMConv3d": spconv.SubMConv3d,
        "SparseConv3d": spconv.SparseConv3d,
    }
    if conv_type not in conv_classes:
        raise ValueError(f"当前编码器不支持稀疏卷积类型: {conv_type}")

    kernel_size = _reverse_3d(kernel_size)
    stride = _reverse_3d(stride)
    padding = _reverse_3d(padding)

    layers = []
    for layer in order:
        if layer == "conv":
            layers.append(
                conv_classes[conv_type](
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key,
                )
            )
        elif layer == "norm":
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            layers.append(nn.ReLU(inplace=True))
    return spconv.SparseSequential(*layers)


class SparseBasicBlockSpconv2(spconv.SparseModule):
    """由两个 spconv2 子流形卷积组成的稀疏残差块。"""

    expansion = 1

    def __init__(self, channels, norm_cfg, indice_key):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = _replace_feature(out, self.relu(self.norm1(out.features)))
        out = self.conv2(out)
        out = _replace_feature(out, self.norm2(out.features))
        out = _replace_feature(out, self.relu(out.features + identity))
        return out


@BACKBONES.register_module()
class SparseEncoderSpconv2(nn.Module):
    """保持原 SparseEncoder 配置接口和 BEV 输出接口的 spconv2 版本。

    DAOcc 体素化 CUDA 算子输出坐标 ``[batch, x, y, z]``，旧版配置的
    ``sparse_shape`` 也是 ``[X, Y, Z]``。spconv2 要求坐标和空间形状为
    ``[batch, z, y, x]`` 与 ``[Z, Y, X]``，转换只在本类内部执行。
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        conv_out_stride=(1, 1, 2),
    ):
        super().__init__()
        if block_type not in ("conv_module", "basicblock"):
            raise ValueError(f"不支持的 block_type: {block_type}")
        if not isinstance(order, (list, tuple)) or len(order) != 3:
            raise TypeError("order 必须包含 conv、norm、act 三项")
        if set(order) != {"conv", "norm", "act"}:
            raise ValueError(f"order 必须包含 conv、norm、act，实际为: {order}")

        # 配置继续沿用旧版 [X,Y,Z]；构造 spconv2 张量时使用 [Z,Y,X]。
        self.sparse_shape = tuple(sparse_shape)
        self.sparse_shape_spconv2 = tuple(reversed(sparse_shape))
        self.in_channels = in_channels
        self.order = tuple(order)
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(encoder_channels)
        self.fp16_enabled = False

        input_order = ("conv",) if self.order[0] != "conv" else self.order
        self.conv_input = make_sparse_convmodule_spconv2(
            in_channels,
            base_channels,
            3,
            norm_cfg=norm_cfg,
            padding=1,
            indice_key="subm1",
            conv_type="SubMConv3d",
            order=input_order,
        )

        encoder_out_channels = self._make_encoder_layers(
            norm_cfg, base_channels, block_type
        )
        self.conv_out = make_sparse_convmodule_spconv2(
            encoder_out_channels,
            output_channels,
            kernel_size=(1, 1, 3),
            stride=conv_out_stride,
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    def _make_encoder_layers(self, norm_cfg, in_channels, block_type):
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_sparse_convmodule_spconv2(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_sparse_convmodule_spconv2(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        if in_channels != out_channels:
                            raise ValueError(
                                "残差块输入输出通道不一致，无法直接相加: "
                                f"{in_channels} != {out_channels}"
                            )
                        blocks_list.append(
                            SparseBasicBlockSpconv2(
                                out_channels,
                                norm_cfg=norm_cfg,
                                indice_key=f"subm{i + 1}",
                            )
                        )
                else:
                    blocks_list.append(
                        make_sparse_convmodule_spconv2(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels

            self.encoder_layers.add_module(
                f"encoder_layer{i + 1}", spconv.SparseSequential(*blocks_list)
            )
        return in_channels

    @auto_fp16(apply_to=("voxel_features",))
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """将体素特征编码为与旧实现相同的二维 BEV 特征。"""
        #* 当前spconv2版本在eval模式下无法为“FP16稀疏输入 + FP32权重”
        
        # -----------------------------------------------------------
        # 找到可用的implicit GEMM推理算法。训练阶段继续保留auto_fp16路径；
        # 仅在验证/推理阶段把体素特征恢复为FP32，使输入与权重dtype一致。
        # 此转换发生在@auto_fp16完成参数转换之后，因此能够覆盖其FP16输入。
        if not self.training:
            voxel_features = voxel_features.float()

        coors = coors.int()
        # DAOcc 坐标 [batch,x,y,z] -> spconv2 [batch,z,y,x]。
        coors_spconv2 = coors[:, [0, 3, 2, 1]].contiguous()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features,
            coors_spconv2,
            self.sparse_shape_spconv2,
            int(batch_size),
        )

        x = self.conv_input(input_sp_tensor)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        spatial_features = self.conv_out(x).dense()
        # spconv2 dense() 返回 [N,C,D,H,W]，将高度 D 折叠到通道维。
        n, c, d, h, w = spatial_features.shape
        return spatial_features.reshape(n, c * d, h, w)

