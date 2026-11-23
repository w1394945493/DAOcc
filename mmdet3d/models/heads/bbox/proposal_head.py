"""CenterHead 候选框及来源特征接口；原检测输出和损失接口保持兼容。"""
import torch
import torch.nn.functional as F

from mmdet3d.core import circle_nms, xywhr2xyxyr
from mmdet3d.models.builder import HEADS
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from .centerpoint import CenterHead


@HEADS.register_module()
class ProposalHead(CenterHead):
    def __init__(self, *args, instance_selection=None, multi_point_features=False,
                 learnable_sampling=False, sampling_points=5,
                 roi_features=False, roi_output_size=3, roi_sampling_ratio=2, **kwargs):
        super().__init__(*args, **kwargs)
        #* (11.19) ROI 优先于多点模式；不构建偏移/权重网络，不混入来源网格向量。
        self.roi_features = roi_features
        for value in (roi_output_size, roi_sampling_ratio):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('roi_output_size 和 roi_sampling_ratio 必须为正整数')
        self.roi_output_size = roi_output_size
        self.roi_sampling_ratio = roi_sampling_ratio
        multi_point_features = multi_point_features and not roi_features
        self.multi_point_features = multi_point_features
        #* (11.19) 多点模式下可预测每个实例自己的 K 个偏移；False 保留固定五点。
        if not isinstance(sampling_points, int) or isinstance(sampling_points, bool) or sampling_points < 1:
            raise ValueError('sampling_points 必须是正整数')
        self.sampling_points = sampling_points
        self.offset_predictor = None
        if learnable_sampling and multi_point_features:
            channels = self.shared_conv.conv.out_channels
            self.offset_predictor = torch.nn.Sequential(
                torch.nn.LayerNorm(channels), torch.nn.Linear(channels, 2 * sampling_points))
            # 初始点分散，避免所有点重合后因对称梯度始终学习同一位置。
            if sampling_points == 5:
                initial = torch.tensor([[0., 0.], [-.25, -.25], [.25, -.25],
                                        [-.25, .25], [.25, .25]])
            elif sampling_points == 1:
                initial = torch.zeros(1, 2)
            else:
                angles = torch.arange(sampling_points) * (2 * torch.pi / sampling_points)
                initial = .25 * torch.stack([angles.cos(), angles.sin()], -1)
            torch.nn.init.zeros_(self.offset_predictor[-1].weight)
            with torch.no_grad():
                self.offset_predictor[-1].bias.copy_(torch.atanh(2 * initial).reshape(-1))
        #* (11.19) 仅从采样点构建实例特征：每实例预测 K 个权重，softmax 自适应聚合。
        # 来源向量仅用于预测位置/权重，不再以残差或固定比例混入输出。
        self.sample_weight_predictor = None
        if multi_point_features:
            channels = self.shared_conv.conv.out_channels
            count = sampling_points if learnable_sampling else 5
            self.sample_weight_predictor = torch.nn.Sequential(
                torch.nn.LayerNorm(channels), torch.nn.Linear(channels, count))
            torch.nn.init.zeros_(self.sample_weight_predictor[-1].weight)
            torch.nn.init.zeros_(self.sample_weight_predictor[-1].bias)
        selection = dict(instance_selection or {})
        self.instance_score_threshold = float(selection.pop('score_threshold', 0.3))
        #* (11.23) 可选关闭训练实例分数门槛；默认开启，推理始终保留门槛。
        self.instance_train_score_filter = selection.pop('train_score_filter', True)
        if not isinstance(self.instance_train_score_filter, bool):
            raise ValueError('train_score_filter须为布尔值')
        #* (11.23) 新增可选训练期GT质量筛选；默认score_only保持原训练/推理行为。
        self.instance_train_filter = selection.pop('train_filter', 'score_only')
        self.instance_box_cost_threshold = float(selection.pop('box_cost_threshold', 1.5))
        self.instance_iou_threshold = float(selection.pop('iou_threshold', 0.3))
        self.instance_class_aware = selection.pop('class_aware', True)
        if not 0 <= self.instance_score_threshold <= 1:
            raise ValueError('score_threshold须在[0,1]范围内')
        if self.instance_train_filter not in ('score_only', 'gt_quality'):
            raise ValueError("train_filter须为'score_only'或'gt_quality'")
        if self.instance_box_cost_threshold < 0 or not 0 <= self.instance_iou_threshold <= 1:
            raise ValueError('box_cost_threshold须非负，iou_threshold须在[0,1]范围内')
        if not isinstance(self.instance_class_aware, bool) or selection:
            raise ValueError('class_aware须为布尔值，instance_selection含未知配置项')
    
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
        # 空候选或关闭注入时仍让偏移参数接入检测图，DDP 获得零梯度；不改变预测数值。
        for predictor in (self.offset_predictor, self.sample_weight_predictor):
            if predictor is not None:
                source = source + sum(p.reshape(-1)[0] * 0 for p in predictor.parameters())
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
            #*================================================#
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
        #! 只跳过训练实例专用阈值；上游检测阈值、NMS及后续GT几何筛选仍然有效。
        apply_score = not self.training or self.instance_train_score_filter
        return ids[scores[ids] >= self.instance_score_threshold] if for_instances and apply_score else ids

    @torch.no_grad()
    def _filter_instances_by_gt(self, item, meta, gt_bboxes, gt_labels):
        """训练时按同类别 GT 质量过滤；GT 只决定保留与否，不替换预测框。"""
        #* (11.23) 复用StreamOcc思路：box_cost<阈值或3D IoU>阈值即可保留。
        count = len(item['boxes'])
        if count == 0:
            return item
        if gt_bboxes is None or gt_labels is None or len(gt_bboxes) == 0:
            return {key: value[:0] for key, value in item.items()}
        device = item['boxes'].device
        gt_bboxes = gt_bboxes.to(device)
        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        if len(gt_bboxes) != len(gt_labels):
            raise ValueError('GT框与GT类别数量不一致')

        pred = item['boxes'].detach().float()
        gt_center = gt_bboxes.gravity_center.float()
        gt_dims = gt_bboxes.dims.float().clamp_min(1e-6)
        # 对应本地 StreamOcc：0.25 * (2*中心L1 + 0.5*log尺寸L1)。
        center_cost = (pred[:, None, :3] - gt_center[None]).abs().sum(-1)
        size_cost = (pred[:, None, 3:6].clamp_min(1e-6).log()
                     - gt_dims[None].log()).abs().sum(-1)
        box_cost = .25 * (2. * center_cost + .5 * size_cost)

        # 解码框的 z 是重心高度；LiDARInstance3DBoxes 需要底面中心高度。
        pred_bottom = pred.clone()
        pred_bottom[:, 2] -= pred_bottom[:, 5] * .5
        box_type = meta['box_type_3d']
        pred_boxes = box_type(pred_bottom, box_dim=pred_bottom.shape[-1])
        iou = pred_boxes.overlaps(pred_boxes, gt_bboxes).to(device)

        if self.instance_class_aware:
            same_class = item['labels'][:, None] == gt_labels[None]
            box_cost = box_cost.masked_fill(~same_class, float('inf'))
            iou = iou.masked_fill(~same_class, -1.)
        keep = ((box_cost.amin(dim=1) < self.instance_box_cost_threshold)
                | (iou.amax(dim=1) > self.instance_iou_threshold))
        return {key: value[keep] for key, value in item.items()}

    def build_instances(self, predictions, source, metas, return_detections=False,
                        gt_bboxes_3d=None, gt_labels_3d=None):
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
        use_gt_filter = self.training and self.instance_train_filter == 'gt_quality'
        #* (11.23) GT筛选严格限制在训练期；推理仍沿用分数阈值、NMS和范围过滤。
        if use_gt_filter and (gt_bboxes_3d is None or gt_labels_3d is None
                              or len(gt_bboxes_3d) != batch or len(gt_labels_3d) != batch):
            raise ValueError('gt_quality训练筛选需要与batch对应的GT框和GT类别')
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
                    if not self.training or self.instance_train_score_filter:
                        valid &= scores[b, keep] >= self.instance_score_threshold
                    keep = keep[valid]
                
                #* 3. 保存筛选后的实例及其来源索引，确保框、类别、特征来源逐行对应。
                # offset 将各任务内部类别编号转换为全局检测类别编号。
                rows[b].append(dict(boxes=boxes[b, keep], scores=scores[b, keep],
                                    labels=labels[b, keep] + offset, source_indices=indices[b, keep],
                                    task_ids=torch.full_like(indices[b, keep], task)))
            offset += self.num_classes[task]
        #* 4. 实例特征三选一：ROI 区域特征、多点加权特征、原来源网格特征。
        # 展平顺序与解码得到的 source_indices 一致：[B,C,X,Y] → [B,X*Y,C]。
        flat = source.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        result = []
        for b, tasks in enumerate(rows):
            item = {key: torch.cat([task[key] for task in tasks]) for key in tasks[0]}
            #! (11.23) 所有任务均未选中框时，cat 仍得到形状合法的空 Tensor；
            #! 后续继续构造空实例字典，不在这里报错或伪造实例。
            # 实例跨任务按分数排序；检测输出仍保留任务拼接顺序，两路行号不保证对应。
            order = item['scores'].argsort(descending=True)
            item = {key: value[order] for key, value in item.items()}
            if use_gt_filter:
                # 分数阈值和NMS已在上游完成；这里只进一步检查预测框的GT几何质量。
                item = self._filter_instances_by_gt(
                    item, metas[b], gt_bboxes_3d[b], gt_labels_3d[b])
            # 几何解码/筛选无梯度，但这里保留 source 的梯度，允许占据损失回传检测特征。
            # 占据范围交集与融合实例数量上限由后续 BoxBEVFusion.build_edges 处理。
            item['features'] = flat[b, item['source_indices']]
            #* (11.19) 多点模式仅聚合采样特征；来源向量只用于预测采样位置与权重。
            # 不改变检测框/分数/NMS；框坐标 detach，但可学习偏移保留梯度。
            # 占据监督通过 grid_sample 同时训练偏移网络和检测来源特征。
            if self.roi_features:
                item['features'], valid_rois = self._roi_instance_features(source[b:b+1], item['boxes'])
                # 无有效采样区域的框仅从注入实例中移除，已保存的检测结果不变。
                item = {key: value[valid_rois] for key, value in item.items()}
            elif self.multi_point_features:
                item['features'] = self._sample_instance_features(source[b:b+1], item)
                # 全部采样越界的实例不参与注入；不退回来源特征，检测输出不受影响。
                valid_samples = item.pop('sampling_valid')
                item = {key: value[valid_samples] for key, value in item.items()}
            result.append(item)
        #* 5. 返回注入实例；推理时同时返回之前保留的完整检测结果。
        # format_detections 仅整理输出格式、将中心 z 转为底面 z，不重复执行后处理。
        # instances 保留中心 z 的框 Tensor 及特征；detections 为框对象、分数、类别。
        if return_detections:
            detections = self.format_detections(detection_rows, metas)
            return result, detections
        return result

    def _roi_instance_features(self, source, boxes):
        """旋转 ROI 对齐：每 bin 固定 r×r 双线性采样，bin 均值后空间平均。

        用 grid_sample 实现，不依赖额外旋转 ROI CUDA 扩展。沿用本检测头解码
        坐标约定（整数索引对应来源格），不额外套用图像 ROI 的半像素平移。
        边界策略：越界采样点不参与 bin 平均，空 bin 不参与空间平均。
        因此不宣称与采用零填充等边界策略的第三方算子逐元素相同。
        """
        n, channels = len(boxes), source.shape[1]
        if n == 0:
            return source.permute(0, 2, 3, 1).reshape(-1, channels)[:0], boxes.new_zeros((0,), dtype=torch.bool)
        with torch.cuda.amp.autocast(enabled=False):
            #* 1. 框仅定义采样区域，detach 断开坐标梯度；保留 source 的特征梯度。
            boxes = boxes.detach().float()
            size, ratio = self.roi_output_size, self.roi_sampling_ratio
            steps = size * ratio
            axis = (torch.arange(steps, device=source.device, dtype=torch.float32) + .5) / steps - .5
            y, x = torch.meshgrid(axis, axis, indexing='ij')
            offsets = torch.stack([x, y], -1).reshape(-1, 2)
            local = boxes[:, None, 3:5] * offsets
            c, sn = boxes[:, 6:7].cos(), boxes[:, 6:7].sin()
            xy = torch.stack([c * local[..., 0] - sn * local[..., 1],
                              sn * local[..., 0] + c * local[..., 1]], -1) + boxes[:, None, :2]
            coder = self.bbox_coder
            ij = (xy - xy.new_tensor(coder.pc_range[:2])) / (xy.new_tensor(coder.voxel_size[:2]) * coder.out_size_factor)
            h, w = source.shape[-2:]
            valid = torch.isfinite(ij).all(-1)
            valid &= (ij[..., 0] >= 0) & (ij[..., 0] <= w - 1)
            valid &= (ij[..., 1] >= 0) & (ij[..., 1] <= h - 1)
            valid &= (torch.isfinite(boxes).all(-1) & (boxes[:, 3:6] > 0).all(-1))[:, None]
            grid = 2 * (ij + .5) / ij.new_tensor([w, h]) - 1
            grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
            #* 2. 一次批量采样，不逐框复制整张源特征图；N×(size*ratio)^2 个查询。
            samples = F.grid_sample(source.float(), grid[None], align_corners=False)[0].permute(1, 2, 0)
            samples = torch.where(valid[..., None], samples, torch.zeros_like(samples))
            samples = samples.reshape(n, size, ratio, size, ratio, channels)
            mask = valid.reshape(n, size, ratio, size, ratio)
            counts = mask.sum((2, 4))
            #* 3. 得到 [N,size,size,C] ROI 特征，再对有效 bin 做平均，输出 [N,C]。
            roi = samples.sum((2, 4)) / counts.clamp_min(1)[..., None]
            bin_valid = counts > 0
            pooled = roi.sum((1, 2)) / bin_valid.sum((1, 2)).clamp_min(1)[:, None]
            return pooled.to(source.dtype), bin_valid.flatten(1).any(-1)

    def _sample_instance_features(self, source, item):
        """按解码公式反求特征图索引；grid_sample 的 x 对应列，y 对应行。"""
        if not len(item['boxes']):
            item['sampling_valid'] = item['scores'].new_zeros((0,), dtype=torch.bool)
            return item['features']
        with torch.cuda.amp.autocast(enabled=False):
            boxes = item['boxes'].detach().float()
            if self.offset_predictor is None:
                # 原固定五点路径保留，sampling_points 仅控制可学习路径。
                offsets = boxes.new_tensor([[0., 0.], [-.25, -.25], [.25, -.25],
                                            [-.25, .25], [.25, .25]])
            else:
                #* (11.19) 来源特征 → 每实例 K×2 偏移；0.5*tanh 限制在框局部范围。
                # 这里不能 detach 偏移，否则占据损失不能学习采样位置。
                offsets = .5 * self.offset_predictor(item['features'].float()).reshape(-1, self.sampling_points, 2).tanh()
            local = boxes[:, None, 3:5] * offsets
            c, sn = boxes[:, 6].cos()[:, None], boxes[:, 6].sin()[:, None]
            xy = torch.stack([c * local[..., 0] - sn * local[..., 1],
                              sn * local[..., 0] + c * local[..., 1]], -1) + boxes[:, None, :2]
            coder = self.bbox_coder
            step = xy.new_tensor(coder.voxel_size[:2]) * coder.out_size_factor
            ij = (xy - xy.new_tensor(coder.pc_range[:2])) / step
            h, w = source.shape[-2:]
            # 反解 _decode_task 的坐标，索引0即来源特征第0个元素，不额外减半格。
            valid = torch.isfinite(ij).all(-1)
            valid &= (ij[..., 0] >= 0) & (ij[..., 0] <= w - 1)
            valid &= (ij[..., 1] >= 0) & (ij[..., 1] <= h - 1)
            grid = 2 * (ij + .5) / ij.new_tensor([w, h]) - 1
            grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
            samples = F.grid_sample(source.float(), grid[None], align_corners=False)[0].permute(1, 2, 0)
            # 旧实现（保留说明）：pooled=有效点平均；输出=0.5*来源+0.5*pooled。
            #* (11.19) 新实现：只对有效采样点归一化；初始等权，随后由占据损失学习权重。
            logits = self.sample_weight_predictor(item['features'].float())
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            weights = logits.softmax(-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            item['sampling_valid'] = valid.any(-1)
            # 全无效时 weights 全0，避免全 -inf softmax 的 NaN；上层再移除此实例。
            return (samples * weights[..., None]).sum(1).to(source.dtype)

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
