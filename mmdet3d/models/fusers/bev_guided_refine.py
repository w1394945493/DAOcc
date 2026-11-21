"""Observation-change guidance for the original BEV RefineNet attention."""
import math
import torch
from torch import nn
from .bev_refine import BEVRefineNet


class BEVGuidedRefineNet(BEVRefineNet):
    """Keep history repair intact; learn additive channel/spatial attention logits.

    The caller prepares history exactly as for BEVRefineNet. Invalid pixels
    suppress new guidance, rather than replacing the history repair output.
    Spatial convolutions may propagate guidance across the validity boundary.
    """

    def __init__(self, channels, internal_channels, guide_channels=64, time_scale=1.0):
        if not isinstance(guide_channels, int) or guide_channels <= 0:
            raise ValueError('guide_channels must be a positive integer')
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be finite and positive')
        super().__init__(channels, internal_channels)
        self.time_scale = time_scale
        groups = math.gcd(guide_channels, 16)
        self.guide = nn.Sequential(
            nn.Conv2d(2 * channels + 2, guide_channels, 1),
            nn.GroupNorm(groups, guide_channels), nn.SiLU(),
            nn.Conv2d(guide_channels, guide_channels, 3, padding=1),
            nn.GroupNorm(groups, guide_channels), nn.SiLU())
        self.channel_bias = nn.Conv2d(guide_channels, channels, 1)
        self.spatial_bias = nn.Conv2d(guide_channels, 1, 1)
        self.init_guidance()

    def init_guidance(self):
        """Call again after any enclosing Xavier initialization, before loading weights."""
        for head in (self.channel_bias, self.spatial_bias):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, current, history, observation, delta_t, valid_mask):
        #* ============ 1. 输入含义与本模块职责 ======================
        # current = F_t：[B,C,X,Y]，当前帧尚未时序融合的观测特征。
        # history = H̃：[B,C,X,Y]，调用方按原 RefineNet 规则准备的历史：
        #   有历史时为 warp 后记忆；首帧使用当前特征；无效槽位按原规则替换；
        #   warp 越界仍保留零填充，不在本模块中强制回退 current。
        # observation = F̃：[B,C,X,Y]，上一帧融合前观测经同一网格 warp 后的结果。
        #   它与 current 属于同一处理阶段，不是融合后的历史记忆 history。
        # delta_t：[B]，当前与缓存观测的实际时间差（秒），反序可为负。
        # valid_mask：[B,X,Y]，历史槽位有效性与 warp 有效区域的交集。
        #   它不是运动标签或置信度；仅控制新增引导，不关闭原历史修复主干。
        # 输出：[B,C,X,Y]，修复后的历史 H^-；后续 Cat 融合和缓存写入由外部完成。
        # 真实历史/观测缓存已在调用方 detach；本方法不会主动截断 current 梯度。
        if observation is None or delta_t is None or valid_mask is None:
            raise ValueError('guided_refine requires observation, delta_t and valid_mask')

        #* ============ 2. 原 RefineNet 生成历史修复内容 ======================
        # R = Bottleneck(H̃)：保留原 1×1→3×3→1×1、BN/ReLU 结构。
        # R 才是待加入历史的修复内容；新增 guide 只调制注意力，不替代 R。
        residual = self.bottleneck(history)
        mask = valid_mask[:, None].bool()  # [B,1,X,Y]，可沿通道维广播。

        #* ============ 3. 构造观测引导输入（FP32） ======================
        # ΔF = F_t - F̃：同阶段观测差分，避免混入“观测→融合记忆”的表征差异。
        # 无效区域的 current、ΔF、Δt 全部清零，避免把 warp 零填充误当场景变化。
        # mask 作为独立通道区分“没有历史”和“存在历史但观测变化为零”。
        # 时间差仅作为条件，不将整个修复残差乘 Δt，也不计算 ΔF/Δt。
        # 仅此处差分/拼接和后面的池化统计显式使用 FP32；不是整个模块强制 FP32。
        # float()/to() 保留梯度连接，当前帧仍能通过引导分支接受监督。
        with torch.cuda.amp.autocast(enabled=False):
            current_fp32 = current.float()
            current_valid = torch.where(mask, current_fp32, 0.)
            delta = torch.where(mask, current_fp32 - observation.float(), 0.)
            dt = delta_t.float().reshape(-1, 1, 1, 1) / self.time_scale
            dt = torch.where(mask, dt.expand(-1, 1, *current.shape[-2:]), 0.)
            guide_input = torch.cat((current_valid, delta, dt, mask.float()), dim=1)
            # guide_input：[B,2C+2,X,Y]，依次为当前观测、差分、时间、有效性。

        #* ============ 4. 提取低维引导特征 ======================
        # E = Guide(input)：1×1→GN→SiLU→3×3→GN→SiLU，输出 [B,G,X,Y]。
        # G 为 guide_channels，默认 64。输入转为卷积权重 dtype，兼容显式 half；
        # 实际卷积精度同时受外层 autocast 策略控制。
        # 输入清零不保证 E 在无效位置为零：卷积、bias 和 GN 都可能产生响应；
        # 因此下面仍需对池化和输出偏置显式使用 mask。
        guide = self.guide(guide_input.to(self.guide[0].weight.dtype))

        #* ============ 5. 生成通道偏置与空间偏置 ======================
        # 通道偏置 cb：[B,C,1,1]，先对 E 的有效历史位置做平均，再映射到 C 通道。
        # count：[B,1,1,1]；FP32 累加避免低精度下大范围求和带来的数值问题。
        # clamp_min(1) 防止首帧除零；count=0 时再次清零 cb，屏蔽输出层 bias。
        with torch.cuda.amp.autocast(enabled=False):
            count = mask.float().sum((2, 3), keepdim=True)
            pooled = (guide.float() * mask).sum((2, 3), keepdim=True) / count.clamp_min(1)
        cb = self.channel_bias(pooled.to(guide.dtype))
        cb = torch.where(count > 0, cb, 0.)
        # 空间偏置 sb：[B,1,X,Y]，逐位置生成；无效位置不直接添加空间引导。
        # cb/sb 是可正可负的 logits 偏置，不是概率或经过标定的可靠性分数。
        sb = torch.where(mask, self.spatial_bias(guide), 0.)

        #* ============ 6. 在原通道注意力上加入观测引导 ======================
        # ℓ_c(R) = MLP(AvgPool(R)) + MLP(MaxPool(R))，两次调用共享参数。
        # 有效位置：a_c = sigmoid(ℓ_c + cb)；无效位置：a_c = sigmoid(ℓ_c)。
        # 正偏置提高对应通道权重，负偏置降低权重；不直接改变 R 的生成方式。
        # cb 在空间上共享，但经 mask 选择后 weights 广播为 [B,C,X,Y]。
        logits = (self.channel_attention(self.avg_pool(residual)) +
                  self.channel_attention(self.max_pool(residual)))
        weights = torch.where(mask, (logits + cb.to(logits.dtype)).sigmoid(), logits.sigmoid())
        residual = residual * weights  # R_c：先完成通道调制，保持原串行顺序。

        #* ============ 7. 在原空间注意力上加入观测引导 ======================
        # 从 R_c 沿通道求均值/最大值，得到 [B,2,X,Y]，再经原 7×7 卷积。
        # a_s = sigmoid(ℓ_s(R_c) + sb)，输出 [B,1,X,Y]，在通道维广播。
        # 注意：sb 在无效区为零，不等于该区输出完全不受邻域引导影响；
        # 因为 7×7 卷积读取的 R_c 邻域可能包含已被引导调制的有效位置。
        spatial = torch.cat((residual.mean(1, keepdim=True), residual.max(1, keepdim=True)[0]), dim=1)
        logits = self.spatial_attention(spatial)
        residual = residual * (logits + sb.to(logits.dtype)).sigmoid()

        #* ============ 8. 保留原残差连接与 ReLU，输出修复历史 ======================
        # H^- = ReLU(H̃ + R_c ⊙ a_s)。不使用旧 controlled 模式的 update_scale。
        # 两个引导输出头零初始化时 cb=sb=0；在主干权重、输入和精度相同的条件下，
        # 本模块恢复原 RefineNet 的前向行为，而不是退化为历史恒等传递。
        # 首帧/全无有效历史时仍执行引导分支，参数保留在计算图中，但引导梯度为零；
        # 原修复主干仍正常工作。有效历史下，零初始化初期输出头先获得梯度，
        # 待其权重非零后，梯度才会进一步传到 guide 骨干。
        return torch.relu(history + residual)
