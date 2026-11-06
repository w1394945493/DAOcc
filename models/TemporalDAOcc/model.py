"""DAOcc 时序变体：原单帧主干不变，occupancy 与 3D 检测分支共享时序融合 BEV。"""
import torch
import numpy as np
from mmcv.runner import auto_fp16
from mmdet3d.models.builder import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion
# （temporal) 新增的对齐函数和融合层，原 BEVFusion 不包含它们。
from .fusion import TemporalFusion, warp_bev


@FUSIONMODELS.register_module()
class TemporalDAOcc(BEVFusion):
    #* （temporal) =================== 新增：时序模块初始化 ===================
    def __init__(self, temporal=None, **kwargs):
        # 先沿用原模型构造相机、LiDAR、Fuser、Decoder 和预测头。
        super().__init__(**kwargs)
        # （temporal) temporal 配置单独解析，不改变原主干的构造参数。
        cfg = dict(temporal or {})
        # （temporal) 此范围对应 Decoder 的原始 LiDAR BEV，不是最终 occupancy 范围。
        self.bev_bounds = cfg.pop('bounds', [-54., -54., 54., 54.])
        self.max_time_gap = cfg.pop('max_time_gap', None)
        self.temporal_fusion = TemporalFusion(**cfg)

    #* ==================== 历史分支：每个槽位重新提取原始 BEV ====================
    def forward(self, *args, history=None, temporal_meta=None, **kwargs):
        if history is not None:
            histories = history if isinstance(history, (list, tuple)) else [history]
            if len(histories) != self.temporal_fusion.num_history:
                raise ValueError('数据集历史数量与 model.temporal.num_history 不一致')
            previous = []
            states = [(module, module.training) for module in self.modules()]
            try:
                self.eval()
                with torch.no_grad():
                    for item in histories:
                        # 共享当前权重；历史仅执行到 Decoder，不执行时序融合及预测头。
                        previous.append(self.forward(**item, _bev_only=True).detach())
            finally:
                for module, state in states:
                    module.training = state
            kwargs['_histories'] = [dict(bev=bev, augmentation=item['lidar_aug_matrix'],
                meta=[m['history'][i] for m in temporal_meta],
                valid=[m['history_valid'][i] for m in temporal_meta])
                for i, (bev, item) in enumerate(zip(previous, histories))]
        return super().forward(*args, temporal_meta=temporal_meta, **kwargs)

    #* ==================== 每帧独立对齐，然后拼接卷积 ====================
    def _temporal_features(self, current, augmentation, kwargs):
        sink = kwargs.get('_cache_sink')
        if sink is not None:
            # 缓存时序融合前的原始 BEV，绝不缓存递归融合结果。
            sink.update(bev=current.detach(), augmentation=augmentation.detach().clone())
        histories = kwargs.get('_histories', [None]*self.temporal_fusion.num_history)
        aligned_features, masks = [], []
        meta = kwargs.get('temporal_meta')
        for old in histories:
            if old is None:
                aligned_features.append(torch.zeros_like(current))
                masks.append(torch.zeros_like(current[:, :1], dtype=torch.bool))
                continue
            gcur = torch.as_tensor(np.stack([m['lidar2global'] for m in meta]),
                                   device=current.device, dtype=torch.float32)
            gprev = torch.as_tensor(np.stack([m['lidar2global'] for m in old['meta']]),
                                    device=current.device, dtype=torch.float32)
            # 当前增强坐标 -> 世界 -> 历史 LiDAR -> 历史增强坐标，供反向采样。
            transform = old['augmentation'].float() @ torch.inverse(gprev) @ gcur @ torch.inverse(augmentation.float())
            aligned, mask = warp_bev(old['bev'], transform, self.bev_bounds)
            valid = [bool(v) and a['scene'] == b['scene'] and
                     0 < a['timestamp']-b['timestamp'] and
                     (self.max_time_gap is None or a['timestamp']-b['timestamp'] <= self.max_time_gap)
                     for a, b, v in zip(meta, old['meta'], old['valid'])]
            mask = mask & torch.tensor(valid, device=current.device).view(-1, 1, 1, 1)
            aligned_features.append(aligned)
            masks.append(mask)
        return self.temporal_fusion(current, aligned_features, masks)

    #* ==================== 单卡流式推理：原始特征的有限队列 ====================
    @torch.no_grad()
    def forward_stream(self, data, state=None):
        if self.training:
            raise RuntimeError('流式推理前请 model.eval()')
        meta = data['temporal_meta']
        if len(meta) != 1:
            raise ValueError('流式推理仅支持 batch size=1')
        current_meta = meta[0]
        # 换场景、重复帧或时间倒退时清空；历史选择仍严格依照 Dataset 的 token。
        if state is None or state['scene'] != current_meta['scene'] or current_meta['timestamp'] <= state['timestamp']:
            state = dict(scene=current_meta['scene'], timestamp=current_meta['timestamp'], entries={})
        entries = state['entries']
        histories = []
        for old_meta, valid in zip(current_meta['history'], current_meta['history_valid']):
            item = entries.get(old_meta['token']) if valid else None
            if valid and item is None:
                raise RuntimeError('缺少所需历史缓存；请按 dataset.stream_indices 连续处理完整场景')
            histories.append(None if item is None else dict(
                bev=item['bev'], augmentation=item['augmentation'], meta=[item['meta']], valid=[True]))
        sink = {}
        inputs = dict(data)
        inputs.pop('history', None)
        result = self.forward(**inputs, _histories=histories, _cache_sink=sink)
        sink['meta'] = current_meta
        entries[current_meta['token']] = sink
        # 至多保留最大历史偏移数量的原始 BEV；dict 保持插入顺序。
        while len(entries) > max(current_meta['history_offsets']):
            del entries[next(iter(entries))]
        state['timestamp'] = current_meta['timestamp']
        return result, state

    #* =================== 原单帧流程；时序改动在下方逐处标记 ===================
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

        #* （temporal) 新增接入点1：只提取原始 BEV，供历史分支或可视化使用。
        # （temporal) _bev_only 是自定义参数，不是 PyTorch 内置开关；默认 False。
        # （temporal) True 时立即返回 x，后续时序融合和所有预测头均不执行。
        if kwargs.get("_bev_only", False):
            return x
        #* （temporal) 新增接入点2：生成 OCC 与 3D 检测共同使用的最终时序 BEV。
        # （temporal) 对齐、融合只执行一次，两类预测头接收同一份 [B,C,X,Y] 特征；
        # 两项损失均可通过融合层回传当前主干，历史分支仍关闭梯度。
        # （temporal) x 保留给可选 map 分支；没有有效历史时历史槽位置零，仍执行卷积融合。
        temporal_x = self._temporal_features(x, lidar_aug_matrix, kwargs)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    #* （temporal) 训练时检测头也使用时序 BEV；检测监督与损失计算方式保持不变。
                    pred_dict = head(temporal_x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif head_type == "map":
                    losses = head(x, gt_masks_bev)
                elif head_type == "occ":
                    # TODO: [yz] this is so weird!
                    #* （temporal) OCC 训练使用与检测头相同的 temporal_x。
                    # （temporal) 原 occupancy 损失函数不变；梯度可经过融合层回传当前主干。
                    occ_pred = head(temporal_x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
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
                    #* （temporal) 普通评估和流式推理均使用最终时序 BEV，与训练一致。
                    pred_dict = head(temporal_x, metas)
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
                    #* （temporal) OCC 推理使用与检测头相同的 temporal_x，解码输出格式保持不变。
                    occ_pred = head(temporal_x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
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

