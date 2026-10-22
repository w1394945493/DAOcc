"""不依赖 mmdet3d 的 BEV 对齐和轻量融合模块，便于单独测试、导出。"""
import torch
from torch import nn
from torch.nn import functional as F


def warp_bev(previous, current_to_previous, bounds):
    # * DAOcc decoder 输出空间轴为 X、Y；grid_sample 要求 Y、X。
    b, c, nx, ny = previous.shape
    xmin, ymin, xmax, ymax = bounds
    x = xmin + (torch.arange(nx, device=previous.device).float() + .5) * (xmax-xmin)/nx
    y = ymin + (torch.arange(ny, device=previous.device).float() + .5) * (ymax-ymin)/ny
    xx, yy = torch.meshgrid(x, y)
    points = torch.stack((xx, yy, torch.zeros_like(xx), torch.ones_like(xx)), -1)
    transformed = torch.einsum('bij,xyj->bxyi', current_to_previous.float(), points)
    gx = 2 * (transformed[..., 0]-xmin)/(xmax-xmin)-1
    gy = 2 * (transformed[..., 1]-ymin)/(ymax-ymin)-1
    # 仅使用历史体素中心覆盖的区域，屏蔽边界零填充混入。
    valid = ((gx.abs() <= 1-1/nx+1e-6) & (gy.abs() <= 1-1/ny+1e-6)).unsqueeze(1)
    grid = torch.stack((gx, gy), -1).transpose(1, 2)
    aligned = F.grid_sample(previous.float().transpose(2, 3), grid,
                            align_corners=False, padding_mode='zeros').transpose(2, 3)
    return aligned.to(previous.dtype), valid


class TemporalFusion(nn.Module):
    def __init__(self, channels=512, mode='gate', alpha=.5):
        super().__init__()
        if mode not in ('off', 'fixed', 'gate') or not 0 <= alpha <= 1:
            raise ValueError('mode 应为 off/fixed/gate，alpha 应在 [0,1]')
        self.mode, self.alpha = mode, alpha
        if mode == 'gate':
            # * 每个空间位置一个门值，仅增加一个 1×1 卷积，初始偏向当前帧。
            self.gate = nn.Conv2d(channels*2, 1, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.)

    def forward(self, current, aligned, valid):
        if self.mode == 'off':
            return current
        weight = self.alpha if self.mode == 'fixed' else torch.sigmoid(
            self.gate(torch.cat((current, aligned), dim=1)))
        # * 无效位置严格返回当前特征，不把零填充作为真实历史参与融合。
        return torch.where(valid, current + weight*(aligned-current), current)
