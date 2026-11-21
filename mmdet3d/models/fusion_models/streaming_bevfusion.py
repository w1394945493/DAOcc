"""DAOcc streaming adapter with optional StreamOcc-style BEV aggregation."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from mmdet3d.models.builder import FUSIONMODELS
#*-----------------------------------------------------------#
#* (11.12) BEV 缓存与对齐组件统一归入模型 utils。
from mmdet3d.models.fusers.bev_history_fusion import BEVHistoryFusion
from mmdet3d.models.utils import BEVFeatureCache, BEVFeatureWarp
from .bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class StreamingBEVFusion(BEVFusion):
    def __init__(self, cache_enabled=True, bev_warp=None, history_fusion=None,
                 fuse_before_decoder=False, **kwargs):
        #* (11.15) 可选 decoder 前/后时序处理；默认保持原有 decoder 后路径。
        # 缓存、对齐与融合均在同一位置执行，避免跨阶段混用历史特征。
        if not isinstance(fuse_before_decoder, bool):
            raise TypeError('fuse_before_decoder must be True or False')
        super().__init__(**kwargs)
        self.fuse_before_decoder = fuse_before_decoder
        self.bev_cache = BEVFeatureCache(enabled=cache_enabled)
        self.bev_warp = BEVFeatureWarp(**bev_warp) if bev_warp is not None else None
        #* (11.12) 可选二维 RefineNet 与拼接融合；关闭时保留仅缓存路径。
        if history_fusion is not None and (not cache_enabled or self.bev_warp is None):
            raise ValueError('History fusion requires enabled cache and BEV warp')
        self.history_fusion = BEVHistoryFusion(**history_fusion) if history_fusion is not None else None
        if self.history_fusion is not None:
            self.bev_warp.mask_outside = False
        self._stream_eval_enabled = False
        self._stream_metas = None
        self._pending_observation = None
        self._pending_bev = None
        # Only alive during forward; process_history and visualization can read the mask/grid.
        self._history_alignment = None

    def reset_stream(self):
        self.bev_cache.reset()
        self._history_alignment = None

    @contextmanager
    def validation_stream(self):
        #* (11.12) DAOCC 使用当前训练权重评估；独立验证缓存替代 StreamOcc 的独立 EMA 对象。
        # 每轮验证从空缓存开始，结束或异常时恢复训练缓存，不重新计算或拷贝训练 BEV。
        saved = (self.bev_cache, self._stream_eval_enabled, self._stream_metas,
                 self._pending_bev, self._history_alignment, self._pending_observation)
        self.bev_cache = BEVFeatureCache(enabled=saved[0].enabled)
        self._stream_eval_enabled = True
        self._stream_metas = None
        self._pending_observation = None
        self._pending_bev = None
        self._history_alignment = None
        try:
            yield
        finally:
            self.bev_cache.reset()
            (self.bev_cache, self._stream_eval_enabled,
             self._stream_metas, self._pending_bev, self._history_alignment, self._pending_observation) = saved

    def forward_single(self, *args, stream_metas=None, **kwargs):
        if not self.training and not self._stream_eval_enabled:
            # 未显式进入流式验证上下文时，仍保留普通单帧推理行为。
            return super().forward_single(*args, **kwargs)
        if stream_metas is None:
            raise ValueError('Streaming forward requires stream_metas from sequence dataset')
        self._stream_metas = stream_metas
        try:
            result = super().forward_single(*args, **kwargs)
            if self._pending_bev is None:
                raise RuntimeError('Streaming stage did not produce a BEV feature')
            self.bev_cache.update(self._pending_bev, stream_metas, self._pending_observation)
            return result
        finally:
            self._stream_metas = None
            self._pending_observation = None
            self._pending_bev = None
            self._history_alignment = None

    def process_bev_features(self, current_bev):
        streaming = self.training or self._stream_eval_enabled
        history_bev, history_valid = self.bev_cache.get(self._stream_metas) if streaming else (None, None)
        if streaming:
            #!-----------------------------------------------------------------------#
            #* (11.15) 固定融合位置下，历史和当前 BEV 的 shape/device 应保持一致。
            # 不兼容说明缓存或前向流程异常，必须报错，不能静默退化为逐帧训练。
            # AMP 允许历史 FP32、当前 FP16；dtype 不同本身不构成异常。
            if history_bev is not None and (history_bev.shape != current_bev.shape or history_bev.device != current_bev.device):
                raise RuntimeError('历史与当前 BEV 的 shape 或 device 不一致')
            # Detached view is only held for this forward; cache.update makes the owned copy.
            self._pending_bev = current_bev.detach()
            #* (11.17) 仅新路径保存融合前观测；成功前向后与融合状态共同写入缓存。
            if self.history_fusion is not None and self.history_fusion.requires_observation:
                self._pending_observation = current_bev.detach()
            #* (11.12) 在原始缓存快照上执行历史→当前 BEV 对齐，不覆盖缓存中的原始 BEV。
            # process_history 接收对齐结果；像素有效掩码、逆向变换和网格存于临时结果中。
            self._history_alignment = None
            if self.bev_warp is not None and history_bev is not None:
                self._history_alignment = self.bev_warp(
                    history_bev, self.bev_cache.metas, self._stream_metas, history_valid)
                history_bev = self._history_alignment['bev']
            output = self.process_history(current_bev, history_bev, history_valid)
            #* (11.12) 缓存融合后的特征，递归积累历史，但不跨帧传播梯度。
            self._pending_bev = output.detach()
            return output
        # 融合模型在普通单帧推理时仍使用首帧路径，但不读写流式缓存。
        return self.process_history(current_bev, None, None)

    def process_history(self, current_bev, history_bev, history_valid):
        """Refine aligned history, concatenate with current, then fuse."""
        if self.history_fusion is None:
            return current_bev
        #*--------------------------------------------------------------#
        #* (11.17) 新路径：两份历史共用 warp 网格和槽位有效性。
        if self.history_fusion.requires_observation:
            batch, _, nx, ny = current_bev.shape
            mask = torch.zeros((batch, nx, ny), device=current_bev.device, dtype=torch.bool)
            dt = torch.zeros(batch, device=current_bev.device, dtype=torch.float32)
            observation = current_bev
            if history_bev is None:
                history_bev = current_bev
            elif self._history_alignment is not None:
                previous = self.bev_cache.observation
                if previous is None or previous.shape != current_bev.shape or previous.device != current_bev.device:
                    raise RuntimeError('History guidance/update requires matching observation cache')
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    observation = F.grid_sample(previous.float(), self._history_alignment['grid'],
                                                mode='bilinear', padding_mode='zeros', align_corners=True)
                mask = self._history_alignment['valid_mask']
                dt = torch.tensor([cur['timestamp'] - old['timestamp'] if valid else 0.
                                   for cur, old, valid in zip(self._stream_metas, self.bev_cache.metas,
                                                              history_valid.tolist())],
                                  device=current_bev.device, dtype=torch.float32)
            if self.history_fusion.mode == 'guided_refine' and history_valid is not None:
                # Same slot fallback and detached replacement as original RefineNet;
                # keep warp zero padding. Pixel validity only gates new guidance.
                replacement = current_bev.detach() if self.training else current_bev
                history_bev = torch.where(history_valid[:, None, None, None], history_bev, replacement)
            return self.history_fusion(current_bev, history_bev, observation, dt, mask)
        #*--------------------------------------------------------------#
        #* (11.12) 参照 StreamOcc：首帧用当前特征；无效历史槽位单独替换。
        # 越界零填充照常参与融合，不使用空间掩码强制回退当前特征。
        if history_bev is None:
            history_bev = current_bev.clone()
        else:
            replacement = current_bev.detach() if self.training else current_bev
            history_bev = torch.where(history_valid[:, None, None, None], history_bev, replacement)
        
        return self.history_fusion(current_bev, history_bev)
