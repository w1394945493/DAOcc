"""Three prediction branches copied from GaussianLSS/model/backbones/agp.py."""
from typing import Optional
import torch
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
    """Predict features, opacity and geometry logits for LiDAR Gaussians."""

    def __init__(self, in_channels=[256], interm_c=128, out_c=128,
                 initial_scale=(0.3, 0.3, 0.3),
                 scale_min=(0.08, 0.08, 0.08),
                 scale_max=(0.64, 0.64, 0.64), learn_geometry=False):
        super().__init__()
        self.learn_geometry = learn_geometry
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
        
        #*============================================#
        if learn_geometry:
            # Geometry attributes share one feature extractor but use independent
            # output layers because their activations and initial values differ.
            self.geometry = nn.Sequential(
                nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(interm_c),
                nn.ReLU(inplace=True),
                BottleneckBlock(interm_c),
                BottleneckBlock(interm_c),
            )
            self.offset = nn.Conv2d(interm_c, 3, kernel_size=1)
            self.scale = nn.Conv2d(interm_c, 3, kernel_size=1)
            self.rotation = nn.Conv2d(interm_c, 4, kernel_size=1)

            initial = torch.as_tensor(initial_scale, dtype=torch.float32)
            lower = torch.as_tensor(scale_min, dtype=torch.float32)
            upper = torch.as_tensor(scale_max, dtype=torch.float32)
            if initial.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
                raise ValueError("initial_scale/scale_min/scale_max must have three values")
            if not torch.all((lower > 0) & (lower < initial) & (initial < upper)):
                raise ValueError("scale bounds must satisfy 0 < min < initial < max")
            # Scale is decoded by log-space sigmoid interpolation. This bias makes
            # the zero-weight head emit exactly initial_scale (0.3 m by default).
            ratio = (initial.log() - lower.log()) / (upper.log() - lower.log())
            scale_bias = torch.logit(ratio)
            for head in (self.offset, self.scale, self.rotation):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            with torch.no_grad():
                self.scale.bias.copy_(scale_bias)

    def forward(self, x):
        if not self.learn_geometry:
            return self.feats(x), self.opacity(x).sigmoid()
        geometry = self.geometry(x)
        return (self.feats(x), self.opacity(x).sigmoid(),
                self.offset(geometry), self.scale(geometry),
                self.rotation(geometry))

