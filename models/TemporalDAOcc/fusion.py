"""不依赖 mmdet3d 的 BEV 对齐和轻量融合模块，便于单独测试、导出。"""
import torch
from torch import nn
from torch.nn import functional as F


#* （temporal) =================== 新增：历史 BEV 运动补偿 ===================
def warp_bev(previous, current_to_previous, bounds):
    #* （temporal) 输入历史特征 [B,C,X,Y]；grid_sample 要求空间轴为 Y、X。
    b, c, nx, ny = previous.shape
    xmin, ymin, xmax, ymax = bounds
    # （temporal) 按实际 BEV 尺寸建立当前帧网格，每个坐标取体素中心而非网格边缘。
    x = xmin + (torch.arange(nx, device=previous.device).float() + .5) * (xmax-xmin)/nx
    y = ymin + (torch.arange(ny, device=previous.device).float() + .5) * (ymax-ymin)/ny
    xx, yy = torch.meshgrid(x, y)
    # （temporal) 使用 z=0 平面的齐次坐标 [x,y,0,1]，这是二维 BEV 的对齐近似。
    points = torch.stack((xx, yy, torch.zeros_like(xx), torch.ones_like(xx)), -1)
    transformed = torch.einsum('bij,xyj->bxyi', current_to_previous.float(), points)
    # （temporal) 将历史物理坐标转换为 grid_sample 要求的 [-1,1] 归一化坐标。
    gx = 2 * (transformed[..., 0]-xmin)/(xmax-xmin)-1
    gy = 2 * (transformed[..., 1]-ymin)/(ymax-ymin)-1
    #* （temporal) 只保留历史体素中心覆盖的区域，避免边缘插值混入零填充值。
    valid = ((gx.abs() <= 1-1/nx+1e-6) & (gy.abs() <= 1-1/ny+1e-6)).unsqueeze(1)
    grid = torch.stack((gx, gy), -1).transpose(1, 2)
    # （temporal) 用 FP32 双线性采样；align_corners=False 与体素中心坐标定义一致。
    # （temporal) 采样后恢复 X、Y 轴顺序和输入精度，返回对齐特征及有效位置掩码。
    aligned = F.grid_sample(previous.float().transpose(2, 3), grid,
                            align_corners=False, padding_mode='zeros').transpose(2, 3)
    return aligned.to(previous.dtype), valid


#* （temporal) =================== 新增：当前与历史 BEV 的轻量融合 ===================
class TemporalFusion(nn.Module):
    def __init__(self, channels=512, mode='ema', alpha=.5):
        super().__init__()
        if mode not in ('off', 'fixed', 'gate', 'ema') or not 0 <= alpha <= 1:
            raise ValueError('mode 应为 off/fixed/gate/ema，alpha 应在 [0,1]')
        self.mode, self.alpha = mode, alpha
        #* （temporal) ema 不创建任何卷积或 Parameter；alpha 是固定的历史保留系数。
        # （temporal) 原 gate 作为显式可选方案保留，只有选择 gate 才创建可学习参数。
        if mode == 'gate':
            #* （temporal) 两帧按通道拼接成 2C，1×1 卷积输出每个位置的一个历史权重。
            # （temporal) 所有通道共享该位置门值；零权重、bias=-2 使初始 sigmoid 约为0.119。
            # （temporal) 因而初始更偏向当前帧；新增参数需经过训练才具有学习到的选择能力。
            self.gate = nn.Conv2d(channels*2, 1, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.)

    def forward(self, current, aligned, valid):
        # （temporal) aligned 必须已完成坐标对齐；本层不负责位姿变换。
        if self.mode == 'off':
            return current

        #* （temporal) EMA：M_t = (1-alpha)*F_t + alpha*warp(M_{t-1})。
        # （temporal) fixed 与 ema 单步公式相同；ema 在线缓存融合结果，fixed 缓存原始特征。
        weight = self.alpha if self.mode in ('fixed', 'ema') else torch.sigmoid(self.gate(torch.cat((current, aligned), dim=1)))

        #* （temporal) 有效位置：(1-w)×当前 + w×对齐历史；无效位置严格返回当前特征。
        # （temporal) 不是拼接后增加整套 Decoder；输出通道数和空间大小保持不变。
        return torch.where(valid, current + weight*(aligned-current), current)
