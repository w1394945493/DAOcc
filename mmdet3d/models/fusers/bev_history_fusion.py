"""2D adaptation of StreamOcc RefineNet and concatenation aggregation.

No auxiliary geometry/semantic heads. Input and output use [B,C,X,Y].
"""
import torch
from torch import nn
from .bev_state_update import BEVStateUpdate


class BEVRefineNet(nn.Module):
    def __init__(self, channels, internal_channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1),
            nn.BatchNorm2d(internal_channels), 
            nn.ReLU(),
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1),
            nn.BatchNorm2d(internal_channels), 
            nn.ReLU(),
            nn.Conv2d(internal_channels, channels, 1),
            nn.BatchNorm2d(channels))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1, bias=False))
        self.spatial_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, history):
        residual = self.bottleneck(history)
        weights = (self.channel_attention(self.avg_pool(residual))
                   + self.channel_attention(self.max_pool(residual))).sigmoid()
        residual = residual * weights
        spatial = torch.cat([residual.mean(dim=1, keepdim=True),
                             residual.max(dim=1, keepdim=True)[0]], dim=1)
        residual = residual * self.spatial_attention(spatial).sigmoid()
        return torch.relu(history + residual)


class BEVHistoryFusion(nn.Module):
    def __init__(self, channels, internal_channels, use_refine=True,
                 use_controlled_update=False, update_scale=0.1, time_scale=1.0):
        super().__init__()
        if channels < 16 or channels % 16 or internal_channels <= 0:
            raise ValueError('channels must be divisible by 16; internal_channels must be positive')
        #* (11.16) 默认先细化历史再融合；关闭时直接使用已对齐的历史特征。
        # Identity 不创建细化参数，避免直接融合实验中出现未参与前向的可训练参数。
        #* (11.17) 新路径替代 RefineNet；开关关闭时保持旧参数名与初始化顺序。
        self.use_controlled_update = use_controlled_update
        if use_controlled_update:
            self.refine_net = nn.Identity()
            self.state_update = BEVStateUpdate(channels, internal_channels, update_scale, time_scale)
        else:
            self.refine_net = BEVRefineNet(channels, internal_channels) if use_refine else nn.Identity()
        self.cat_block = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GroupNorm(16, channels), nn.ReLU(inplace=True))
        # StreamAgg.init_weights uses Xavier for all multidimensional parameters.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, current, history, observation=None, delta_t=None, valid_mask=None):
        #* (11.18) 1. 输入含义：特征均为 [B,C,X,Y]，处于当前帧增强后的 LiDAR 坐标系。
        # current：当前帧观测 F_t，尚未经过本次时序融合。
        #   decoder 前融合时，来自当前帧多模态融合输出；后融合时来自 decoder 输出。
        # history：已对齐的历史记忆 W(H_{t-1})；H_{t-1} 是缓存帧的时序融合输出。
        #   不是上一帧的原始观测；首帧由调用方以 current 代替。
        # observation：已对齐的历史观测 W(F_{t-1})，不是当前观测。
        #   F_{t-1} 是缓存帧在同一融合位置、尚未时序融合的特征；用于计算
        #   delta_F = current - observation。首帧以 current 代替，使差分为零。
        # delta_t：[B]，当前时间戳减缓存时间戳，单位秒；按实际帧间隔计算，
        #   跳帧会增大间隔，反序为负，无有效历史的槽位为零；不是固定 0.5 秒。
        # valid_mask：[B,X,Y] 的布尔掩码，结合槽位历史有效性和 warp 网格边界。
        #   True 表示存在有效历史采样支持，不代表动态目标或语义置信度；
        #   False 位置在 state_update 内清零控制增量，并将历史输出回退 current。
        # observation、delta_t、valid_mask 仅受控更新路径使用，由调用方提供；
        # 原 RefineNet/直接拼接路径无需这三个参数，允许保持 None。
        # 两份历史均来自 detach 缓存，current 在训练中保留梯度。
        
        #* 2. 历史处理：受控更新替代 RefineNet；旧路径 use_refine=False 时为 Identity。
        if self.use_controlled_update:
            history = self.state_update(current, history, observation, delta_t, valid_mask)
            # 显式统一新路径 Cat 输入为 FP32；旧路径维持原行为。
            current = current.float()
        else:
            history = self.refine_net(history)
        #* 3. 共用 Cat 融合：历史在前、当前在后，2C 通道映射回 C。
        # 返回 [B,C,X,Y] 的更新记忆 H_t，由流式模型写入缓存，并继续后续预测。
        return self.cat_block(torch.cat([history, current], dim=1))
