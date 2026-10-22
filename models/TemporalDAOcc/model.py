"""DAOcc 时序变体：原单帧主干不变，仅 occupancy 分支融合历史 BEV。"""
import torch
import numpy as np
from mmcv.runner import auto_fp16
from mmdet3d.models.builder import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion
from .fusion import TemporalFusion, warp_bev


@FUSIONMODELS.register_module()
class TemporalDAOcc(BEVFusion):
    def __init__(self, temporal=None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(temporal or {})
        self.bev_bounds = cfg.pop('bounds', [-54., -54., 54., 54.])
        self.max_time_gap = cfg.pop('max_time_gap', 1.)
        self.temporal_fusion = TemporalFusion(**cfg)

    def forward(self, *args, history=None, temporal_meta=None, **kwargs):
        if history is not None and self.temporal_fusion.mode != 'off':
            # * 历史分支不反传梯度，也不更新 BN/Dropout 状态；之后恢复各模块状态。
            states = [(module, module.training) for module in self.modules()]
            try:
                self.eval()
                with torch.no_grad():
                    previous = self.forward(**history, _bev_only=True)
            finally:
                for module, state in states:
                    module.training = state
            kwargs['_history_bev'] = previous
            kwargs['_history_aug'] = history['lidar_aug_matrix']
            kwargs['_history_meta'] = [m['history'] for m in temporal_meta]
            kwargs['_history_valid'] = [m['history_valid'] for m in temporal_meta]
        return super().forward(*args, temporal_meta=temporal_meta, **kwargs)

    def _temporal_features(self, current, augmentation, kwargs):
        sink = kwargs.get('_cache_sink')
        if sink is not None:
            # 缓存原始当前 BEV，避免递归累计融合误差；不保留训练计算图。
            sink['bev'] = current.detach()
            sink['augmentation'] = augmentation.detach().clone()
        previous = kwargs.get('_history_bev')
        if previous is None or self.temporal_fusion.mode == 'off':
            return current
        meta, old_meta = kwargs['temporal_meta'], kwargs['_history_meta']
        device = current.device
        gcur = torch.as_tensor(np.stack([m['lidar2global'] for m in meta]),
                              device=device, dtype=torch.float32)
        gprev = torch.as_tensor(np.stack([m['lidar2global'] for m in old_meta]),
                               device=device, dtype=torch.float32)
        # * 将当前增强后的 LiDAR 坐标映射到历史增强后的 LiDAR 坐标，再反向采样。
        transform = (kwargs['_history_aug'].float() @ torch.inverse(gprev)
                     @ gcur @ torch.inverse(augmentation.float()))
        aligned, mask = warp_bev(previous, transform, self.bev_bounds)
        valid = [bool(v) and a['scene'] == b['scene']
                 and 0 < a['timestamp']-b['timestamp'] <= self.max_time_gap
                 for a, b, v in zip(meta, old_meta, kwargs['_history_valid'])]
        mask = mask & torch.tensor(valid, device=device).view(-1, 1, 1, 1)
        return self.temporal_fusion(current, aligned, mask)

    @torch.no_grad()
    def forward_stream(self, data, state=None):
        # * 显式传入/返回缓存；首帧、定位重置时传 None。在线接口限定 batch=1。
        if self.training:
            raise RuntimeError('流式推理前请 model.eval()')
        meta = data['temporal_meta']
        if len(meta) != 1:
            raise ValueError('流式推理仅支持 batch size=1')
        data = dict(data)
        data.pop('history', None)
        sink = {}
        if state is not None:
            data.update(_history_bev=state['bev'], _history_aug=state['augmentation'],
                        _history_meta=state['meta'], _history_valid=[True])
        result = self.forward(**data, _cache_sink=sink)
        sink['meta'] = meta
        return result, sink

    # * 以下保留原 forward_single 的主干及检测损失流程。
    # 原实现仍完整保留在 mmdet3d/models/fusion_models/bevfusion.py。
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    **kwargs,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        if self.decoder is not None:
            x = self.decoder["backbone"](x)
            x = self.decoder["neck"](x)

        # 历史分支只计算到 decoder，不执行预测头。
        if kwargs.get("_bev_only", False):
            return x
        occ_x = self._temporal_features(x, lidar_aug_matrix, kwargs)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif head_type == "map":
                    losses = head(x, gt_masks_bev)
                elif head_type == "occ":
                    # TODO: [yz] this is so weird!
                    occ_pred = head(occ_x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
                    losses = head.loss(occ_pred, kwargs['voxel_semantics'], kwargs['mask_camera'])
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{head_type}/{name}"] = val * self.loss_scale[head_type]
                    else:
                        outputs[f"stats/{head_type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif head_type == "occ":
                    # TODO: [yz] this is so weird!
                    occ_pred = head(occ_x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
                    occ_pred = head.get_occ(occ_pred)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "occ_pred": occ_pred[k],  # already in cpu
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {head_type}")
            return outputs

