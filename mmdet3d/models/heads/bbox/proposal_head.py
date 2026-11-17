"""CenterHead 候选框及来源特征接口；原检测输出和损失接口保持兼容。"""
import torch

from mmdet3d.core import circle_nms, xywhr2xyxyr
from mmdet3d.models.builder import HEADS
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from .centerpoint import CenterHead


@HEADS.register_module()
class ProposalHead(CenterHead):
    def __init__(self, *args, instance_selection=None, **kwargs):
        super().__init__(*args, **kwargs)
        selection = dict(instance_selection or {})
        self.instance_score_threshold = float(selection.pop('score_threshold', 0.3))
        if selection or not 0 <= self.instance_score_threshold <= 1:
            raise ValueError('instance_selection 仅支持 [0,1] 范围的 score_threshold')
    
    # 为什么额外写这个类？因为实例融合需要原检测接口没有返回的两项信息：
    # 1. 检测中间特征 source
    # 用它提取每个预测实例的特征向量，并保留占据损失向检测特征的梯度。
    # 2. 预测框对应的来源网格索引 source_indices
    # 原来的检测后处理返回框、分数和类别，没有保留“这个框最初来自哪个热图网格”。新接口需要让这个索引经过 Top-K、过滤、NMS 后仍与框对应。
    def forward_with_features(self, feats, metas=None):
        """一次检测前向，同时暴露 [B,C,X,Y] 来源特征，不使用 hooks/缓存。"""
        if isinstance(feats, (list, tuple)):
            if len(feats) != 1:
                raise ValueError('ProposalHead 仅支持单尺度检测 BEV')
            feats = feats[0]
        source = self.shared_conv(feats) # 原有共享卷积
        # 与父类 multi_apply(forward_single, [feats]) 的任务/尺度嵌套保持一致。
        predictions = tuple([task(source)] for task in self.task_heads) # 原有各任务预测分支
        
        # 检测网络层、参数和预测计算复用原来的实现。
        # 新增接口额外返回 shared_conv 的中间特征 source，用于按候选来源索引提取实例向量。
        return predictions, source

    @torch.no_grad()
    def _decode_task(self, pred):
        """保留 Top-K 空间索引；解码后的 z 为几何中心，不是框底面。"""
        coder = self.bbox_coder
        heat = pred['heatmap'].detach().float().sigmoid()
        k = min(coder.max_num, heat.shape[-2] * heat.shape[-1])
        # A1. 与可视化 decode_candidates 一致：每个任务、每个样本从分类热图选 Top-K。
        # indices:[B,K] 是来源网格的展平编号，决定之后从 shared_conv 的哪里提取实例特征。
        scores, indices, labels, ys, xs = coder._topk(heat, K=k)
        def gather(name):
            return coder._transpose_and_gather_feat(pred[name].detach().float(), indices)
        # 同一 indices 提取中心偏移、尺寸、朝向等回归量，保证框与来源特征有共同身份。
        # 最终框中心经过偏移回归，因此不能用框中心取整来反推原来的 indices。
        reg = gather('reg')
        x = (xs[..., None] + reg[..., :1]) * coder.out_size_factor * coder.voxel_size[0] + coder.pc_range[0]
        y = (ys[..., None] + reg[..., 1:2]) * coder.out_size_factor * coder.voxel_size[1] + coder.pc_range[1]
        dims = gather('dim')
        if self.norm_bbox:
            dims = dims.exp()
        rot = gather('rot')
        parts = [x, y, gather('height'), dims, torch.atan2(rot[..., :1], rot[..., 1:2])]
        if 'vel' in pred:
            parts.append(gather('vel'))
        return torch.cat(parts, -1), scores, labels.long(), indices

    @torch.no_grad()
    def _select_task(self, boxes, scores, labels, task, meta, for_instances=True):
        """使用原配置和 NMS 算子，返回原候选行号；所有筛选只作用于本样本。"""
        cfg, coder = self.test_cfg, self.bbox_coder
        #* (11.16) 共用输出路径按原检测范围/阈值筛选，不提前施加实例专用规则。
        valid = torch.ones_like(scores, dtype=torch.bool)
        if for_instances:
            valid &= torch.isfinite(boxes).all(-1) & torch.isfinite(scores) & (boxes[:, 3:6] > 0).all(-1)
        if coder.post_center_range is not None:
            bounds = boxes.new_tensor(coder.post_center_range)
            valid &= ((boxes[:, :3] >= bounds[:3]) & (boxes[:, :3] <= bounds[3:])).all(-1)
        if coder.score_threshold:
            valid &= scores > coder.score_threshold
        # A2. ids 始终是当前任务原候选的行号；后续过滤/NMS都更新它，而非丢掉来源关系。
        ids = valid.nonzero(as_tuple=True)[0]
        nms_type = cfg['nms_type'][task] if isinstance(cfg['nms_type'], list) else cfg['nms_type']
        if nms_type == 'circle':
            if ids.numel():
                keep = circle_nms(torch.cat([boxes[ids, :2], scores[ids, None]], -1).cpu().numpy(),
                                  cfg['min_radius'][task], post_max_size=cfg['post_max_size'])
                ids = ids[torch.as_tensor(keep, device=ids.device, dtype=torch.long)]
        elif nms_type == 'rotate':
            if cfg['score_threshold'] > 0:
                ids = ids[scores[ids] >= cfg['score_threshold']]
            if ids.numel():
                bev = meta['box_type_3d'](boxes[ids], coder.code_size).bev
                scales = cfg.get('nms_scale', 1.0)
                scales = scales[task] if isinstance(scales, list) else [scales] * self.num_classes[task]
                for label, scale in enumerate(scales):
                    mask = labels[ids] == label
                    bev[mask, 2:4] *= scale
                keep = nms_gpu(xywhr2xyxyr(bev), scores[ids], cfg['nms_thr'],
                               pre_maxsize=cfg['pre_max_size'], post_max_size=cfg['post_max_size'])
                ids = ids[keep]
                limits = cfg.get('post_center_limit_range')
                if limits:
                    bounds = boxes.new_tensor(limits)
                    centers = boxes[ids, :3]
                    ids = ids[((centers >= bounds[:3]) & (centers <= bounds[3:])).all(-1)]
        else:
            raise ValueError(f'Unsupported NMS type: {nms_type}')
        # 此阈值是预测分类分数门槛，不是与GT的IoU或特征相似度阈值。
        # 占据区域交集及实例数上限留给 BoxBEVFusion.build_edges，避免无交集框占配额。
        return ids[scores[ids] >= self.instance_score_threshold] if for_instances else ids

    def build_instances(self, predictions, source, metas, return_detections=False):
        """几何选择无梯度；最后 gather 在 no_grad 外，允许占据损失回传检测特征。"""
        #* (11.16) 设计目的：共用一次解码/NMS，同时保留完整检测输出和占据注入实例。
        # detections 用于推理输出/评估；instances 用于占据融合，额外携带来源网格特征。
        # 推理时按框集合有 instances ⊆ detections（可以相等）：先保存完整检测结果，
        # 再从中筛选注入实例，避免实例专用条件删掉本应参与检测评估的框。
        # 训练只需要 instances；检测 loss 直接使用 predictions，不使用筛选后的框。
        # return_detections 控制是否同时返回最终检测结果，不控制是否构建实例。
        batch, channels = source.shape[:2]
        if len(metas) != batch:
            raise ValueError('metas 与检测特征 batch 不一致')
        rows = [[] for _ in range(batch)]
        detection_rows = [[] for _ in range(batch)]
        offset = 0
        for task, levels in enumerate(predictions):
            if len(levels) != 1:
                raise ValueError('仅支持单尺度候选')
            #* 1. 解码检测候选，同时保留其来源网格索引，供筛选后提取实例特征。
            boxes, scores, labels, indices = self._decode_task(levels[0])
            for b in range(batch):
                #* 2. 根据是否需要完整检测输出，选择实例筛选的执行位置。
                # 推理 return_detections=True：for_instances=False，仅做检测筛选/NMS；
                # 实例专用条件留到保存检测结果之后执行，保证 instances 是 detections 的子集。
                # 训练 return_detections=False：for_instances=True，在 _select_task 内直接
                # 完成实例筛选，无需保留完整检测输出。其有效性检查在 NMS 前，分数筛选在后。
                keep = self._select_task(boxes[b], scores[b], labels[b], task, metas[b],
                                         for_instances=not return_detections)
                if return_detections:
                    # 保留未施加实例专用条件的检测结果，供最终输出；后续不再解码或做 NMS。
                    detection_rows[b].append((boxes[b, keep], scores[b, keep], labels[b, keep] + offset))
                    # 从检测结果中进一步选择注入实例：数值有限、尺寸为正、达到实例分数阈值。
                    # 仅缩小实例集合，不改变上方保存的检测集合。
                    valid = torch.isfinite(boxes[b, keep]).all(-1) & torch.isfinite(scores[b, keep])
                    valid &= (boxes[b, keep, 3:6] > 0).all(-1)
                    valid &= scores[b, keep] >= self.instance_score_threshold
                    keep = keep[valid]
                
                #* 3. 保存筛选后的实例及其来源索引，确保框、类别、特征来源逐行对应。
                # offset 将各任务内部类别编号转换为全局检测类别编号。
                rows[b].append(dict(boxes=boxes[b, keep], scores=scores[b, keep],
                                    labels=labels[b, keep] + offset, source_indices=indices[b, keep],
                                    task_ids=torch.full_like(indices[b, keep], task)))
            offset += self.num_classes[task]
        #* 4. 按来源网格提取实例特征，用于占据融合，而非对框内区域做 ROI 池化。
        # 展平顺序与解码得到的 source_indices 一致：[B,C,X,Y] → [B,X*Y,C]。
        flat = source.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        result = []
        for b, tasks in enumerate(rows):
            item = {key: torch.cat([task[key] for task in tasks]) for key in tasks[0]}
            # 实例跨任务按分数排序；检测输出仍保留任务拼接顺序，两路行号不保证对应。
            order = item['scores'].argsort(descending=True)
            item = {key: value[order] for key, value in item.items()}
            # 几何解码/筛选无梯度，但这里保留 source 的梯度，允许占据损失回传检测特征。
            # 占据范围交集与融合实例数量上限由后续 BoxBEVFusion.build_edges 处理。
            item['features'] = flat[b, item['source_indices']]
            result.append(item)
        #* 5. 返回注入实例；推理时同时返回之前保留的完整检测结果。
        # format_detections 仅整理输出格式、将中心 z 转为底面 z，不重复执行后处理。
        # instances 保留中心 z 的框 Tensor 及特征；detections 为框对象、分数、类别。
        if return_detections:
            detections = self.format_detections(detection_rows, metas)
            return result, detections
        return result

    @torch.no_grad()
    def format_detections(self, rows, metas):
        """合并已筛选结果；不再次解码或执行 NMS。"""
        detections = []
        for tasks, meta in zip(rows, metas):
            boxes = torch.cat([task[0] for task in tasks]).clone()
            scores = torch.cat([task[1] for task in tasks])
            labels = torch.cat([task[2].int() for task in tasks])
            # 与原 get_bboxes 一致：几何中心 z 转框底面 z；不修改注入分支的框。
            boxes[:, 2] -= boxes[:, 5] * 0.5
            detections.append([meta['box_type_3d'](boxes, self.bbox_coder.code_size), scores, labels])
        return detections
