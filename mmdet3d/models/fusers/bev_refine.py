"""Shared history-only BEV refinement backbone."""
import torch
from torch import nn


class BEVRefineNet(nn.Module):
    # 原历史修复模块：仅根据输入历史生成修复内容，不直接读取当前观测或时间差。
    # BEVGuidedRefineNet 复用这些层，并在注意力 logits 上额外加入观测引导。
    def __init__(self, channels, internal_channels):
        super().__init__()
        #* ============ 1. 历史修复瓶颈：降维 → 空间处理 → 升维 ======================
        # channels=C 为输入/输出通道数，internal_channels=r 为瓶颈通道数。
        # 1×1：C→r，混合通道并降维；3×3：r→r，利用局部邻域生成修复内容；
        # 1×1：r→C，恢复通道数。stride=1、padding=1 保持 BEV 空间尺寸不变。
        # 前两层采用 BN+ReLU；末层只有 BN，无末尾 ReLU，因此候选残差可正可负。
        # BN 在训练时使用当前批次统计并更新运行统计，在评估时使用运行统计。
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(),
            nn.Conv2d(internal_channels, channels, 1),
            nn.BatchNorm2d(channels))

        #* ============ 2. 通道注意力：全局描述与共享映射 ======================
        # 两种池化分别将 [B,C,X,Y] 压缩为 [B,C,1,1]：
        # 平均池化描述整体响应，最大池化保留各通道的峰值响应。
        # 两份描述共用同一个 C→C/16→C 映射，参数不因调用两次而增加。
        # 本层输出 logits；相加并 sigmoid 的操作放在 forward 中。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1, bias=False))

        #* ============ 3. 空间注意力：局部位置权重 ======================
        # 输入为通道均值/最大值组成的 [B,2,X,Y]；7×7 卷积输出 [B,1,X,Y]。
        # padding=3 保持尺寸；sigmoid 在 forward 中执行，权重沿通道维共享。
        self.spatial_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, history):
        #* ============ 1. 输入含义与模块边界 ======================
        # history=H̃：[B,C,X,Y]，由调用方准备的历史特征。
        # 正常流式路径为 warp 后的融合记忆；首帧/无效历史槽位由调用方替换，
        # warp 越界的零填充也作为输入参与修复。本模块不单独处理有效性 mask。
        # 不在这里执行 warp、缓存读写或当前帧 Cat 融合，也不主动 detach 输入。
        # 真实历史缓存的跨帧梯度已在外部截断，但本模块参数仍正常接受任务监督。

        #* ============ 2. 从历史本身生成候选修复内容 ======================
        # R = Bottleneck(H̃)，形状仍为 [B,C,X,Y]。
        # 这是要加入历史的候选残差，不是注意力权重；其值允许为正或负。
        residual = self.bottleneck(history)

        #* ============ 3. 先进行通道注意力调制 ======================
        # a_c = sigmoid(MLP(AvgPool(R)) + MLP(MaxPool(R)))。
        # weights：[B,C,1,1]，广播到所有空间位置；每个通道使用一个全局权重。
        # 注意力是学习得到的调制系数，不直接代表语义置信度或目标概率。
        weights = (self.channel_attention(self.avg_pool(residual))
                   + self.channel_attention(self.max_pool(residual))).sigmoid()
        residual = residual * weights  # R_c = R ⊙ a_c。

        #* ============ 4. 再基于通道调制结果生成空间注意力 ======================
        # 对 R_c 沿通道维计算均值和最大值，各得 [B,1,X,Y]，拼成 [B,2,X,Y]。
        # a_s = sigmoid(Conv7×7([Mean_c(R_c), Max_c(R_c)]))。
        # a_s：[B,1,X,Y]，在通道维广播，调节不同位置的修复幅度。
        # 这是串行注意力：空间权重依赖已完成通道调制的 R_c，并非两路独立计算。
        spatial = torch.cat([residual.mean(dim=1, keepdim=True),
                             residual.max(dim=1, keepdim=True)[0]], dim=1)
        residual = residual * self.spatial_attention(spatial).sigmoid()

        #* ============ 5. 残差相加与非负输出 ======================
        # H^- = ReLU(H̃ + R_c ⊙ a_s)，输出形状保持 [B,C,X,Y]。
        # 注意力非负，但残差内容可正可负，因此仍能增强或抑制历史响应；
        # 最后的 ReLU 将相加后的负值截断为零，保留原 RefineNet 的输出约束。
        # H^- 是修复后的历史，尚不是最终缓存；调用方再与当前观测 Cat 融合。
        # 本模块不强制切换到 FP32，计算精度遵循模型 dtype 与外部 autocast 策略。
        return torch.relu(history + residual)
