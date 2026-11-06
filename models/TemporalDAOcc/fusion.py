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
    xx, yy = torch.meshgrid(x, y, indexing="ij")
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


#* ==================== 可学习多帧 BEV 融合 ====================
class TemporalFusion(nn.Module):
    """按固定历史槽位拼接，再学习通道组合和局部空间融合。"""
    def __init__(self, channels=512, num_history=1):
        super().__init__()
        if num_history < 1:
            raise ValueError('num_history 必须至少为1')
        self.num_history = num_history
        # 当前及历史：[B,C,X,Y]；拼接：[B,(N+1)*C,X,Y]；输出：[B,C,X,Y]。
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*(num_history+1), channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, current, histories, masks):
        if len(histories) != self.num_history or len(masks) != self.num_history:
            raise ValueError('历史槽位数必须与 num_history 一致')
        # 无效历史置零；即使首帧全部历史缺失，也经过同一个可学习模块。
        # 使用 where 而非乘零，避免无效区域潜在的 NaN 被保留。
        features = [current] + [torch.where(mask, old, torch.zeros_like(old))
                                for old, mask in zip(histories, masks)]
        return self.fusion(torch.cat(features, dim=1))
