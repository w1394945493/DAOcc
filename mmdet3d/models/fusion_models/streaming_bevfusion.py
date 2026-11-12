"""DAOcc streaming adapter with optional StreamOcc-style BEV aggregation."""
from contextlib import contextmanager
import torch
from mmdet3d.models.modules.bev_history_fusion import BEVHistoryFusion
from mmdet3d.models.builder import FUSIONMODELS
#* (11.12) BEV 缓存与对齐组件统一归入模型 utils。
from mmdet3d.models.utils import BEVFeatureCache, BEVFeatureWarp
from .bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class StreamingBEVFusion(BEVFusion):
    def __init__(self, cache_enabled=True, bev_warp=None, history_fusion=None, **kwargs):
        super().__init__(**kwargs)
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
                 self._pending_bev, self._history_alignment)
        self.bev_cache = BEVFeatureCache(enabled=saved[0].enabled)
        self._stream_eval_enabled = True
        self._stream_metas = None
        self._pending_bev = None
        self._history_alignment = None
        try:
            yield
        finally:
            self.bev_cache.reset()
            (self.bev_cache, self._stream_eval_enabled,
             self._stream_metas, self._pending_bev, self._history_alignment) = saved

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
                raise RuntimeError('Decoder did not produce a BEV feature')
            self.bev_cache.update(self._pending_bev, stream_metas)
            return result
        finally:
            self._stream_metas = None
            self._pending_bev = None
            self._history_alignment = None

    def process_bev_features(self, current_bev):
        streaming = self.training or self._stream_eval_enabled
        history_bev, history_valid = self.bev_cache.get(self._stream_metas) if streaming else (None, None)
        if streaming:
            if history_bev is not None and (history_bev.shape != current_bev.shape
                                           or history_bev.device != current_bev.device
                                           or history_bev.dtype != current_bev.dtype):
                self.bev_cache.reset()
                history_bev, history_valid = None, None
            # Detached view is only held for this forward; cache.update makes the owned copy.
            self._pending_bev = current_bev.detach()
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
        #* (11.12) 参照 StreamOcc：首帧用当前特征；无效历史槽位单独替换。
        # 越界零填充照常参与融合，不使用空间掩码强制回退当前特征。
        if history_bev is None:
            history_bev = current_bev.clone()
        else:
            replacement = current_bev.detach() if self.training else current_bev
            history_bev = torch.where(history_valid[:, None, None, None], history_bev, replacement)
        return self.history_fusion(current_bev, history_bev)
