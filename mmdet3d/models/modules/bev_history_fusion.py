"""2D adaptation of StreamOcc RefineNet and concatenation aggregation.

No auxiliary geometry/semantic heads. Input and output use [B,C,X,Y].
"""
import torch
from torch import nn


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
    def __init__(self, channels, internal_channels):
        super().__init__()
        if channels < 16 or channels % 16 or internal_channels <= 0:
            raise ValueError('channels must be divisible by 16; internal_channels must be positive')
        self.refine_net = BEVRefineNet(channels, internal_channels)
        self.cat_block = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GroupNorm(16, channels), nn.ReLU(inplace=True))
        # StreamAgg.init_weights uses Xavier for all multidimensional parameters.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, current, history):
        return self.cat_block(torch.cat([self.refine_net(history), current], dim=1))
