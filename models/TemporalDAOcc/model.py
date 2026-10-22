"""DAOcc 时序变体：原单帧主干不变，仅 occupancy 分支融合历史 BEV。"""
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
        self.max_time_gap = cfg.pop('max_time_gap', 1.)
        self.temporal_fusion = TemporalFusion(**cfg)

    #* （temporal) =================== 新增：成对输入的历史分支 ===================
    def forward(self, *args, history=None, temporal_meta=None, **kwargs):
        #* （temporal) 数据集可加载多历史，但当前融合头仍只接收一帧历史。
        if isinstance(history, (list, tuple)):
            raise ValueError('数据集已加载多帧历史，但当前模型仅支持单历史融合；'
                             '训练/评估请使用 multi_adj_frame_id_cfg: [1, 2, 1]。'
                             '多帧数据可先通过 test_temporal.py --history-slot 选择可视化。')
        if history is not None and self.temporal_fusion.mode != 'off':
            #* （temporal) 历史分支关闭梯度，使用 eval 防止更新 BN 统计或启用 Dropout。
            # （temporal) 保存各子模块原状态，避免恢复时误将冻结模块切回 train。
            states = [(module, module.training) for module in self.modules()]
            try:
                self.eval()
                with torch.no_grad():
                    # （temporal) 复用同一模型提取历史原始 BEV；history 内没有嵌套 history，
                    # 因此这次调用不会再次进入历史分支；到 Decoder 后即返回。
                    previous = self.forward(**history, _bev_only=True)
            finally:
                # （temporal) 即使提取失败也恢复状态，当前帧仍按原训练/评估状态运行。
                for module, state in states:
                    module.training = state
            # （temporal) 将历史特征、增强矩阵、位姿/时间和有效性传给当前帧融合步骤。
            kwargs['_history_bev'] = previous
            kwargs['_history_aug'] = history['lidar_aug_matrix']
            kwargs['_history_meta'] = [m['history'] for m in temporal_meta]
            kwargs['_history_valid'] = [m['history_valid'] for m in temporal_meta]
        # （temporal) 父类 forward 会调用本类 forward_single，新增参数通过 kwargs 继续传递。
        return super().forward(*args, temporal_meta=temporal_meta, **kwargs)

    #* （temporal) =================== 新增：缓存、对齐、有效性与融合 ===================
    def _temporal_features(self, current, augmentation, kwargs):
        # （temporal) current 是当前帧 Decoder 输出；augmentation 是当前 LiDAR 增强矩阵。
        sink = kwargs.get('_cache_sink')
        if sink is not None:
            # （temporal) 先用当前 BEV 初始化缓存，首帧/关闭融合时直接使用该结果。
            #* （temporal) EMA 模式在下方用融合结果更新缓存，从而递归保留更早历史。
            sink['bev'] = current.detach()
            sink['augmentation'] = augmentation.detach().clone()
        previous = kwargs.get('_history_bev')
        if previous is None or self.temporal_fusion.mode == 'off':
            # （temporal) 没有历史或关闭时序时，原样返回当前 BEV。
            return current
        meta, old_meta = kwargs['temporal_meta'], kwargs['_history_meta']
        device = current.device
        # （temporal) G = ego2global × lidar2ego，将每帧 LiDAR 坐标变换到世界坐标。
        gcur = torch.as_tensor(np.stack([m['lidar2global'] for m in meta]),
                              device=device, dtype=torch.float32)
        gprev = torch.as_tensor(np.stack([m['lidar2global'] for m in old_meta]),
                               device=device, dtype=torch.float32)
        #* （temporal) 反向采样变换：A历史 × G历史逆 × G当前 × A当前逆。
        # （temporal) 从右向左依次撤销当前增强、转到世界、转到历史 LiDAR、施加历史增强。
        # （temporal) 这样可为当前每个网格点找到应从历史 BEV 的哪个位置取值。
        transform = (kwargs['_history_aug'].float() @ torch.inverse(gprev)
                     @ gcur @ torch.inverse(augmentation.float()))
        aligned, mask = warp_bev(previous, transform, self.bev_bounds)
        # （temporal) warp_bev 的 mask 检查空间覆盖；下面再检查整帧的场景/时间有效性。
        valid = [bool(v) and a['scene'] == b['scene']
                 and 0 < a['timestamp']-b['timestamp'] <= self.max_time_gap
                 for a, b, v in zip(meta, old_meta, kwargs['_history_valid'])]
        mask = mask & torch.tensor(valid, device=device).view(-1, 1, 1, 1)
        #* （temporal) 只有历史有效且覆盖当前位置时参与融合，否则保持 current。
        fused = self.temporal_fusion(current, aligned, mask)
        if sink is not None and self.temporal_fusion.mode == 'ema':
            #* （temporal) 缓存 M_t 而非 F_t，这是指数移动平均与普通两帧加权的区别。
            # （temporal) 无效场景/时间/空间位置已回退当前特征，相应位置的旧记忆被重置。
            sink['bev'] = fused.detach()
        return fused

    #* （temporal) =================== 新增：在线缓存推理接口 ===================
    @torch.no_grad()
    def forward_stream(self, data, state=None):
        #* （temporal) state 是上次返回的缓存；首帧/定位重置传 None。仅支持 batch=1。
        if self.training:
            raise RuntimeError('流式推理前请 model.eval()')
        meta = data['temporal_meta']
        if len(meta) != 1:
            raise ValueError('流式推理仅支持 batch size=1')
        data = dict(data)
        # （temporal) 在线模式复用缓存，不再读取并计算成对输入中的历史分支。
        data.pop('history', None)
        sink = {}
        if state is not None:
            data.update(_history_bev=state['bev'], _history_aug=state['augmentation'],
                        _history_meta=state['meta'], _history_valid=[True])
        result = self.forward(**data, _cache_sink=sink)
        # （temporal) _temporal_features 填入 EMA 更新后的 BEV（其他模式为原始 BEV）及增强矩阵。
        # （temporal) 再附上当前位姿等元信息；调用者下一帧将 sink 作为 state 传回。
        sink['meta'] = meta
        return result, sink

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
        #* （temporal) 新增接入点2：生成 occupancy 分支使用的时序 BEV；x 仍保留原始特征。
        occ_x = self._temporal_features(x, lidar_aug_matrix, kwargs)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    # （temporal) 检测分支保持使用当前 x，不切换到 occ_x。
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif head_type == "map":
                    losses = head(x, gt_masks_bev)
                elif head_type == "occ":
                    # TODO: [yz] this is so weird!
                    #* （temporal) 修改接入点3（训练）：原 head(x, ...) 改为 head(occ_x, ...)。
                    # （temporal) 原 occupancy 损失函数不变；梯度可经过融合层回传当前主干。
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
                    #* （temporal) 修改接入点3（推理）：用 occ_x 预测占据，解码输出格式保持不变。
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

