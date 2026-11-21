"""Three prediction branches copied from GaussianLSS/model/backbones/agp.py."""
from typing import Optional
import torch.nn as nn
from torchvision.models.resnet import Bottleneck
BottleneckBlock = lambda x: Bottleneck(x, x // 4)

class PrepareChannel(nn.Module):
    """Transform the feature map to align with Network."""

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        interm_c=128,
        out_c: Optional[int] = 128,
        depth_num=0,
    ):
        super().__init__()
        assert depth_num != 0

        in_c = sum(in_channels)
        self.feats = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, out_c, kernel_size=1, padding=0),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, depth_num, kernel_size=1, padding=0)
        )
        self.opacity = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, 1, kernel_size=1, padding=0)
        )
        
    def forward(self, x):
        return self.feats(x), self.depth(x), self.opacity(x).sigmoid()


class PrepareLidarGaussianChannel(nn.Module):
    """Predict image features and opacity for LiDAR-centred Gaussians."""

    def __init__(self, in_channels=[256], interm_c=128, out_c=128):
        super().__init__()
        in_c = sum(in_channels)
        self.feats = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, out_c, kernel_size=1),
        )
        self.opacity = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.feats(x), self.opacity(x).sigmoid()

