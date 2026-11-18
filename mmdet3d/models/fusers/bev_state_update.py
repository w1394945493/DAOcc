"""Observation-controlled BEV memory update; not explicit physical motion estimation."""
import math
import torch
from torch import nn


class BEVStateUpdate(nn.Module):
    def __init__(self, channels, internal_channels, update_scale=0.1, time_scale=1.0):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be finite and positive')
        if not math.isfinite(update_scale) or update_scale <= 0:
            raise ValueError('update_scale must be finite and positive')
        #* (11.18) 1. 对应公式：V=P([ΔF,Δt/τ₀])，Q=context([H̃,F_t])，
        # M=α·a_c(Q)·a_s(Q)，δH=M·V，H_t^-=H̃+δH（仅有效历史区域）。
        # time_scale 是 τ₀，单位秒；update_scale 是固定 α，训练和推理始终生效，
        # 不是学习率或仅初始化时使用的系数，也不是可学习参数。
        self.time_scale, self.update_scale = time_scale, update_scale
        groups = math.gcd(internal_channels, 16)
        #* 2. 上下文分支 Q：两份 C 通道特征拼接后，以 2C→r→r→C 瓶颈提取上下文。
        # 借鉴 RefineNet 的局部卷积结构，但输入增加当前观测；Q 不直接充当更新量。
        # GN 无跨帧运行统计；gcd 保证分组数能够整除内部通道数。
        self.context = nn.Sequential(
            nn.Conv2d(2 * channels, internal_channels, 1),
            nn.GroupNorm(groups, internal_channels), nn.SiLU(),
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1),
            nn.GroupNorm(groups, internal_channels), nn.SiLU(),
            nn.Conv2d(internal_channels, channels, 1))
        #* 3. 调制分支：复用 RefineNet 的通道/空间注意力思想。
        # 通道分支对空间平均池化、最大池化结果使用同一组映射，随后相加并 sigmoid。
        # 空间分支对 Q 的通道平均值、最大值拼接，再经 7×7 卷积和 sigmoid。
        # 与原 RefineNet 的串行注意力不同，这里两种调制都直接从 Q 生成。
        # 输出是学习到的响应系数，不是经过标定的置信度或动态目标概率。
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1, bias=False))
        self.spatial_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        #* 4. 控制分支 V：C 通道观测增量 + 1 通道时间增量 → C 通道候选更新。
        # V = W_F·ΔF + w_t·(Δt/τ₀)。无 bias/激活/归一化，保持对控制增量的线性映射；
        # 控制输入全零时 V=0，且 V 可正可负。ΔF 已是增量，不再额外乘 Δt。
        self.control = nn.Conv2d(channels + 1, channels, 1, bias=False)

    def forward(self, current, history, observation, delta_t, valid_mask):
        #* (11.18) 1. 输入与符号对应。
        # current = F_t：[B,C,X,Y]，当前帧融合前观测，保留当前帧梯度。
        # history = H̃=W(H_{t-1})：已对齐的历史融合记忆，来源为 detach 缓存。
        # observation = F̃=W(F_{t-1})：已对齐的历史融合前观测，并非当前观测。
        # delta_t：[B]，实际时间戳差（秒）；跳帧使用真实间隔，反序可为负。
        # valid_mask：[B,X,Y]，有效历史槽位与 warp 有效区域的交集。
        # 本模块输出 H_t^-（修正后的历史）；后续 BEVHistoryFusion 再与 F_t Cat 融合，
        # 得到真正写入缓存的 H_t。本模块自身不执行 Cat 融合或缓存写入。
        #
        #* 2. 数值策略：当前实现整个更新分支使用 FP32，而非整个 DAOcc 使用 FP32。
        # float() 不切断梯度；FP32 是保守实现选择，不是受控动力学的数学要求。
        with torch.cuda.amp.autocast(enabled=False):
            current = current.float()
            # [B,1,X,Y] 可在通道维广播。无效历史先回退 current，
            # 避免用越界零填充构造假的观测变化；不影响有效位置的历史来源。
            mask = valid_mask[:, None].bool()
            history = torch.where(mask, history.float(), current)
            observation = torch.where(mask, observation.float(), current)

            #* 3. 构建控制增量 ΔF 与 Δt/τ₀。
            # 比较同阶段的前后观测，而不是 current-history（后者混入融合造成的差异）。
            delta = torch.where(mask, current - observation, torch.zeros_like(current))
            dt = delta_t.float().reshape(-1, 1, 1, 1) / self.time_scale
            # 将每个槽位的时间差广播为空间图，无效位置清零，包括时间驱动项。
            dt = torch.where(mask, dt.expand(-1, 1, *current.shape[-2:]), 0.)

            #* 4. 根据历史记忆与当前观测生成 Q，再生成通道/空间调制。
            q = self.context(torch.cat((history, current), dim=1))  # Q：[B,C,X,Y]
            ca = (self.channel_attention(q.mean((2, 3), keepdim=True)) +
                  self.channel_attention(q.amax((2, 3), keepdim=True))).sigmoid()  # a_c：[B,C,1,1]
            spatial = torch.cat((q.mean(1, keepdim=True), q.amax(1, keepdim=True)), dim=1)
            sa = self.spatial_attention(spatial).sigmoid()  # a_s：[B,1,X,Y]

            #*-------------------------------------------------------------------------#
            #* 5. 因子化受控更新：δH = M ⊙ V，其中 M=α·ca·sa。
            # self.control(...) 就是 V；下面通过广播实现逐通道、逐空间位置调制。
            # 对应 A=M⊙w_t/τ₀、B=Diag(M)W_F，代码不显式构造对角矩阵。
            # 时间项和观测变化项共享 M；这是受限参数化，不是独立的两个 A/B 网络。
            change = self.update_scale * ca * sa * self.control(torch.cat((delta, dt), dim=1))

            #* 6. 有效位置残差更新，无效位置直接返回当前观测。
            # 相加后不做 ReLU，允许状态向正/负方向修正；ΔF=0 时仍可因 Δt≠0 更新。
            # 首帧全 False：输出回退 current，但仍计算掩码分支，参数保留在计算图中，
            # 该分支获得零梯度，避免首帧额外引入 DDP 未使用参数问题。
            return torch.where(mask, history + change, current)
