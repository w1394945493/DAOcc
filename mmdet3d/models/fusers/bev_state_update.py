"""Observation-controlled BEV memory update; not explicit physical motion estimation."""
import math
import torch
from torch import nn


# 本模块是旧 controlled_update 方案，替代原 RefineNet；不是新 guided_refine 方案。
# 有效历史区域：Q=Context([H̃,F_t])，V=P([ΔF,Δt/τ₀])，
# δH=α·a_c(Q)·a_s(Q)·V，H^-=H̃+δH；无效区域回退当前观测。
# 下列 #! 仅标记关键性能疑点，属于待验证机制，不是已确认的退化原因。
# 现有实验同时改变结构、归一化、边界处理等因素，单次结果不能分离各项因果贡献。
class BEVStateUpdate(nn.Module):
    def __init__(self, channels, internal_channels, update_scale=0.1, time_scale=1.0):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be finite and positive')
        if not math.isfinite(update_scale) or update_scale <= 0:
            raise ValueError('update_scale must be finite and positive')
        # ============ 1. 时间尺度与固定更新缩放 ======================
        # time_scale=τ₀，单位秒；update_scale=α，为正的固定系数。
        # α 在整个训练/推理期间生效，不是学习率、初始化系数或可学习参数。
        #! 实验使用 α=0.01；当 ca≈sa≈0.5 时，有效乘数约为 0.0025。
        # 这可能使分支接近历史恒等传递，但实际 δH/H̃ 还取决于特征和投影权重尺度。
        # 应统计有效区域 RMS(change)/RMS(history)，不能仅凭 α 判定分支失效。
        # 梯度也包含该缩放，但 AdamW 的自适应机制使其不能简单等价为学习速度降低百倍。
        self.time_scale, self.update_scale = time_scale, update_scale
        groups = math.gcd(internal_channels, 16)
        # ============ 2. 上下文分支：历史与当前观测共同生成 Q ======================
        # 输入 [B,2C,X,Y]，通过 2C→r→r→C 瓶颈输出 [B,C,X,Y]。
        # GN 无运行统计；gcd(r,16) 确保分组数整除通道数，激活使用 SiLU。
        #! 与原 RefineNet 不同，这里的 3×3 卷积只为注意力提供上下文，
        # Q 不直接作为残差内容加入历史。局部历史修复能力不能仅凭瓶颈外形视为保留。
        # BN/ReLU 改为 GN/SiLU 且取消末层 BN，也改变了分布；不能预先断言 GN 更差。
        self.context = nn.Sequential(
            nn.Conv2d(2 * channels, internal_channels, 1),
            nn.GroupNorm(groups, internal_channels), nn.SiLU(),
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1),
            nn.GroupNorm(groups, internal_channels), nn.SiLU(),
            nn.Conv2d(internal_channels, channels, 1))
        # ============ 3. 通道/空间调制：两路均从 Q 生成 ======================
        # 通道：空间均值和最大值经共享 C→C/16→C 映射，相加后 sigmoid。
        # 空间：对 Q 求通道均值/最大值，拼接后经 7×7 卷积和 sigmoid。
        # 原 RefineNet 是先通道后空间的串行注意力；这里两路直接读取 Q。
        # ca 在所有位置共享，sa 在所有通道共享，乘积形成可分离调制；
        # 难以直接表达“不同位置需要不同通道组合”的任意权重图。
        # 注意力仅表示学习到的响应系数，不应解释为可靠性或动态目标概率。
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1, bias=False))
        self.spatial_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        # ============ 4. 控制投影：由观测增量和时间生成更新内容 ======================
        # V=P([ΔF,Δt/τ₀])=W_F·ΔF+w_t·Δt/τ₀，输出 [B,C,X,Y]。
        # 无 bias/激活/归一化，保证控制输入全零时 V=0；V 本身可正可负。
        #! 真正生成更新内容的仅为逐位置线性 1×1 卷积，不能直接从邻域搬运内容；
        # 对动态位移或 warp 模糊的处理受限，不等价于原 RefineNet 的非线性空间残差。
        # ΔF 属于观测空间，而更新目标是融合记忆；固定线性映射可能不足以适应两者差异。
        # 当 ΔF 很小时，历史仍可能需要修复，此时主要剩下表达受限的时间项。
        self.control = nn.Conv2d(channels + 1, channels, 1, bias=False)

    def forward(self, current, history, observation, delta_t, valid_mask):
        # ============ 1. 输入、输出与梯度边界 ======================
        # current=F_t：[B,C,X,Y]，当前帧融合前观测，保留当前帧梯度。
        # history=H̃=W(H_{t-1})：对齐后的历史融合记忆。
        # observation=F̃=W(F_{t-1})：同一网格对齐的上一帧融合前观测。
        # delta_t：[B]，实际时间差（秒），跳帧增大间隔、反序允许负值。
        # valid_mask：[B,X,Y]，历史槽位有效性与 warp 有效区域的交集。
        # 输出 H^-：修正后的历史；后续 Cat 融合产生真正写入缓存的 H_t。
        # 真实历史/观测已在外部 detach；本模块不执行缓存写入或额外跨帧反传。
        # 与基线相同，跨帧梯度被截断，不能将递归缓存误认为具有长序列 BPTT 监督。

        # ============ 2. FP32 更新与无效历史输入回退 ======================
        # 关闭 autocast 并将特征转为 FP32；float() 不切断梯度。
        # 当前标准路径中分支参数保持 FP32；这里没有主动把已 half 的权重转回 FP32。
        # 这只控制本更新分支，不代表整个 DAOcc 都以 FP32 运行。
        # FP32 主要影响开销，不能据此认定精度退化；它也不保证永不出现非有限值。
        with torch.cuda.amp.autocast(enabled=False):
            current = current.float()
            # mask：[B,1,X,Y]，沿通道广播。无效处用 current 替换两份历史，
            # 避免 current 减越界零填充产生虚假的观测变化。
            mask = valid_mask[:, None].bool()
            history = torch.where(mask, history.float(), current)
            observation = torch.where(mask, observation.float(), current)

            # ============ 3. 构建同阶段观测差分与带符号时间输入 ======================
            # ΔF=F_t-F̃，而不是 F_t-H̃，避免将融合操作引起的表征差异当作观测变化。
            # ΔF 已是区间增量，不再乘 Δt，也不除以 Δt；时间输入单独广播为空间图。
            # 无效处两种控制量都为零；ΔF=0、Δt≠0 时仍可存在时间驱动更新。
            # 同阶段差分仍混合运动、遮挡、采样、增强、插值及训练参数变化，
            # 不是纯运动信号；带符号时间也不意味着模型具有可逆物理动力学。
            delta = torch.where(mask, current - observation, torch.zeros_like(current))
            dt = delta_t.float().reshape(-1, 1, 1, 1) / self.time_scale
            dt = torch.where(mask, dt.expand(-1, 1, *current.shape[-2:]), 0.)

            # ============ 4. 从上下文计算通道与空间调制 ======================
            # Q：[B,C,X,Y]；ca：[B,C,1,1]；sa：[B,1,X,Y]，通过广播共同调制 V。
            # 通道池化并未按 mask 排除无效区域；回退 current 的边界会参与全局统计，
            # 还可能通过 GN/局部卷积影响有效区域。影响大小需结合有效比例验证。
            q = self.context(torch.cat((history, current), dim=1))
            ca = (self.channel_attention(q.mean((2, 3), keepdim=True)) +
                  self.channel_attention(q.amax((2, 3), keepdim=True))).sigmoid()
            spatial = torch.cat((q.mean(1, keepdim=True), q.amax(1, keepdim=True)), dim=1)
            sa = self.spatial_attention(spatial).sigmoid()

            # ============ 5. 因子化受控更新：δH=M⊙V ======================
            # M=α·ca·sa；对应 A=M⊙w_t/τ₀、B=Diag(M)W_F，不显式创建对角矩阵。
            # 时间项与观测项共享 M，这是受限参数化，不是两个独立的 A/B 网络。
            # 固定时间方向下，时间项每通道的符号由 w_t 决定，正值 M 只能调幅；
            # 无法根据上下文任意改变该通道时间更新的方向，两项也可能互相抵消。
            # 可分别统计观测项、时间项及最终 change，区分幅度不足与内容不合适。
            change = self.update_scale * ca * sa * self.control(torch.cat((delta, dt), dim=1))

            # ============ 6. 有效处残差相加，无效处返回当前观测 ======================
            # 有效：H^-=H̃+δH；无效：H^-=F_t。相加后不做原 RefineNet 的 ReLU。
            # 首帧全无效时仍执行分支，参数留在计算图中但获得零梯度，避免未使用参数问题。
            #! 这里替代了原 RefineNet，而不是在其修复结果上叠加更新；
            # 若 change 很弱，等于失去原修复能力，却未获得足够的补偿。
            #! 原基线会对首帧替代特征和越界零填充执行 RefineNet；此处直接回退 current，
            # 同时改变了首帧学习机会、边界分布及进入 Cat 的特征来源。
            # 不做最终 ReLU 也改变输出分布，但允许负值本身不能直接判定为缺陷。
            # 可视化中无效边界的亮条带可能仅来自回退：评估更新强度应只统计有效区域，
            # 用原始张量差而非绝对值均值热图之差，并结合多帧/多种子消融确认原因。
            return torch.where(mask, history + change, current)
