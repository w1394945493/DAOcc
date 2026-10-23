"""将二维 BEV 解码为有界的三维语义高斯；仅依赖 PyTorch。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def grid_centers(bounds, shape):
    #* 所有三维张量统一按 X、Y、Z 排列；取体素中心而不是网格边界。
    lo, hi = torch.tensor(bounds[:3]), torch.tensor(bounds[3:])
    axes = [lo[i] + (torch.arange(n).float() + .5) * (hi[i]-lo[i])/n
            for i, n in enumerate(shape)]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)


def inverse_covariance(scales, quaternion):
    #* 四元数顺序为 w,x,y,z。Sigma=R diag(s²) Rᵀ，其逆可直接构造，无 CPU 求逆。
    q = F.normalize(quaternion, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    matrix = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                          2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                          2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
    rotation = matrix.reshape(*scales.shape[:-1], 3, 3)
    return (rotation * scales.reciprocal().square().unsqueeze(-2)) @ rotation.transpose(-1, -2)


class GaussianPredictor(nn.Module):
    def __init__(self, channels, bounds, anchor_shape=(50, 50, 4),
                 scale_min=.1, scale_max=1.2, scale_init=.6,
                 rotation=True, offset_fraction=.5):
        super().__init__()
        if len(bounds) != 6 or len(anchor_shape) != 3 or min(anchor_shape) < 1:
            raise ValueError('bounds 必须为六维范围，anchor_shape 必须为三个正整数')
        if any(bounds[i+3] <= bounds[i] for i in range(3)):
            raise ValueError('高斯范围上界必须大于下界')
        if not 0 < scale_min < scale_init < scale_max or not 0 < offset_fraction <= 1:
            raise ValueError('要求 0<scale_min<scale_init<scale_max，0<offset_fraction<=1')
        self.anchor_shape = tuple(anchor_shape)
        self.scale_min, self.scale_max = scale_min, scale_max
        self.rotation = rotation
        self.offset_fraction = offset_fraction
        # 每个高斯：3中心偏移 + 3尺度 + 4四元数残差 + 1语义混合权重 + 17语义。
        self.attributes = 28 if rotation else 24
        self.projection = nn.Conv2d(channels, anchor_shape[2]*self.attributes, 1)
        self.register_buffer('anchors', grid_centers(bounds, anchor_shape).reshape(1, -1, 3), persistent=False)
        self.register_buffer('lower', torch.tensor(bounds[:3]).float(), persistent=False)
        self.register_buffer('upper', torch.tensor(bounds[3:]).float(), persistent=False)
        self.register_buffer('step', (self.upper-self.lower)/torch.tensor(anchor_shape), persistent=False)
        # 小权重保留输入梯度；尺度初始化确保初始邻域有覆盖，不能全部落到最小尺度。
        nn.init.normal_(self.projection.weight, std=.001)
        nn.init.zeros_(self.projection.bias)
        p = (scale_init-scale_min)/(scale_max-scale_min)
        with torch.no_grad():
            self.projection.bias.view(anchor_shape[2], self.attributes)[:, 3:6] = math.log(p/(1-p))

    # * 从BEV预测一组语义高斯：200x200 占据区域的BEV特征 → 平均池化 → 50x50 BEV特征 → 1x1卷积 → 每个XY位置预测4组高斯属性 → 50x50x4=10000 个高斯
    # * 采用固定数量的规则锚点 + 可学习偏移 实现
    def forward(self, bev):
        #* =================== 1. 将BEV降采样到高斯锚点的XY网格 ===================
        # 输入来自占据头的坐标变换和卷积，形状为[B,C,Y,X]，不是原始图像。
        # nx、ny是XY方向的锚点数，nz是每个XY位置预设的高度锚点数。
        # 默认50×50×4，共G=10000个高斯；并不是最终200×200×16的占据体素。
        nx, ny, nz = self.anchor_shape
        # 平均池化只改变XY特征分辨率，减少后续属性预测的计算量。
        bev = F.adaptive_avg_pool2d(bev, (ny, nx)) # (1 128 200 200) -> (1 128 50 50)

        #* =================== 2. 预测属性，并整理成逐个高斯的列表 ===================
        # projection是可学习的1×1卷积，每个XY位置一次输出nz组高斯属性。
        # 开启旋转时，每组A=28：中心偏移3 + 尺度3 + 旋转残差4 + 权重1 + 语义17。
        # 默认卷积输出[B,112,50,50]，再拆为[B,4,28,50,50]。
        # 此时raw仍是无约束数值，下面才转换成有物理含义的参数。
        raw = self.projection(bev).float().view(bev.shape[0], nz, self.attributes, ny, nx)
        # [B,Z,A,Y,X] → [B,X,Y,Z,A] → [B,G,A]，排列顺序必须与self.anchors一致。
        raw = raw.permute(0, 4, 3, 1, 2).reshape(bev.shape[0], -1, self.attributes)

        #* =================== 3. 解码三维中心和尺度 ===========================
        # 中心 = 预设锚点 + 学习到的局部偏移；不是直接预测任意绝对坐标。
        # tanh限制偏移幅度，step是XYZ锚点间距；默认最多偏移各方向的半个间距。
        means = self.anchors + raw[..., :3].tanh()*self.step*self.offset_fraction
        # 中心限制在占据范围内部，避免CUDA聚合计算网格索引时越界。
        means = torch.maximum(torch.minimum(means, self.upper-1e-4), self.lower+1e-4)
        # sigmoid将三个尺度限制在配置的正数区间，分别控制高斯局部三轴的扩展程度。
        # 尺度与旋转共同决定椭球形状；尺度不是目标检测框的长宽高。
        scales = self.scale_min + (self.scale_max-self.scale_min)*raw[..., 3:6].sigmoid()

        #* =================== 4. 解码旋转，可选择保持轴对齐 ===================
        # cursor标记后续权重所在的属性位置：无旋转时为6，有旋转时为10。
        cursor = 6
        # 四元数采用[w,x,y,z]；[1,0,0,0]表示不旋转。
        quaternion = raw.new_zeros(*raw.shape[:2], 4)
        quaternion[..., 0] = 1
        if self.rotation:
            # 在单位四元数上加有界残差，避免得到全零四元数。
            # 后面的inverse_covariance会先归一化，再转换为旋转矩阵。
            quaternion = quaternion + .5*raw[..., 6:10].tanh()
            cursor = 10

        #* =================== 5. 解码语义混合权重和非空类别概率 ===================
        #* 此处opacity沿用localagg_prob定义，仅参与多个高斯的语义加权；
        # 它不直接控制占据概率，也不能把它理解为关闭某个高斯的开关。
        opacity = raw[..., cursor].sigmoid()
        # 每个高斯预测17个非空类别的概率，softmax后这17项之和为1。
        semantic = raw[..., cursor+1:].softmax(-1)
        # 原版CUDA内核固定18通道，所以末尾补一列0以匹配接口。
        #* 这里的0不是最终free概率；free由后续aggregation根据几何占据概率计算。
        semantic = F.pad(semantic, (0, 1))

        #* =================== 6. 返回高斯属性，交给后续聚合生成占据结果 ===================
        # 本函数只生成高斯，尚未输出每个体素的类别。
        # means/scales：[B,G,3]；opacity：[B,G]；semantics：[B,G,18]。
        # inverse_covariance：[B,G,3,3]，描述各方向距离对高斯响应的影响。
        # 用尺度和旋转直接构造逆协方差，不需要将矩阵搬到CPU求逆。
        return dict(means=means, scales=scales, opacity=opacity,
                    semantics=semantic, inverse_covariance=inverse_covariance(scales, quaternion))
