"""检测先行的占据融合模型；继承编码器与公共入口，原 BEVFusion 不改动。"""
from mmcv.runner import auto_fp16
from mmdet3d.models.builder import FUSIONMODELS
from .bevfusion import BEVFusion
from ..heads.bbox.proposal_head import ProposalHead
from ..heads.occ.instance_bev_occ_head import InstanceBEVOCCHead2D


@FUSIONMODELS.register_module()
class InstanceBEVFusion(BEVFusion):
    def __init__(self, *args, instance_to_occ=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_to_occ = instance_to_occ
        if 'object' not in self.heads or not isinstance(self.heads['object'], ProposalHead):
            raise ValueError('InstanceBEVFusion 需要 ProposalHead')
        if 'occ' not in self.heads or not isinstance(self.heads['occ'], InstanceBEVOCCHead2D):
            raise ValueError('InstanceBEVFusion 需要 InstanceBEVOCCHead2D')
        if set(self.heads) != {'object', 'occ'}:
            raise ValueError('首版 InstanceBEVFusion 仅支持 object 和 occ 两个任务')

    @auto_fp16(apply_to=('img', 'points'))
    def forward_single(self, img, points, camera2ego, lidar2ego, lidar2camera,
                       lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix,
                       lidar_aug_matrix, metas, gt_masks_bev=None, gt_bboxes_3d=None,
                       gt_labels_3d=None, **kwargs):
        #* (11.16) 训练与推理共用此入口；下面按执行顺序区分原流程与新增接口。
        # 原 DAOcc：相机/LiDAR 编码 -> 多模态 BEV 融合 -> BEV decoder -> 各任务头。
        # 本次改进：保留共享特征提取，任务头改为“先检测，再引导占据预测”。
        #*-----------------------------------------------------------------------------#
        #* 1. 【原 DAOcc】分别提取相机与 LiDAR 的 BEV 特征。
        # 复用父类的编码器和提取方法，网络结构及参数名保持不变。
        # 相机分支包含图像编码与视角变换；LiDAR 分支包含体素化与点云编码。
        # 推理时反向执行两个分支是原有显存安排，不代表融合顺序或空间方向改变。
        features = []
        sensors = list(self.encoders) if self.training else list(self.encoders)[::-1]
        for sensor in sensors:
            if sensor == 'camera':
                feature = self.extract_camera_features(
                    img, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
                    camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix,
                    metas, **kwargs)
            elif sensor == 'lidar':
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f'unsupported sensor: {sensor}')
            features.append(feature)
        
        # 恢复原模态顺序，使训练与推理送入 fuser 的特征排列一致。
        if not self.training:
            features.reverse()
        #*-----------------------------------------------------------------------------#
        #* 2. 【原 DAOcc】融合不同模态的 BEV，得到共享 BEV 特征。
        # self.fuser 是原多模态融合模块，不是历史融合或检测引导占据融合。
        # 无 fuser 时只允许单模态，直接使用该模态特征。
        if self.fuser is not None:
            bev = self.fuser(features)
        else:
            if len(features) != 1:
                raise ValueError('多模态输入需要 fuser')
            bev = features[0]
        #*-----------------------------------------------------------------------------#
        #* 3. 【近期新增：时序扩展接口】可在 decoder 前处理 BEV。
        # 本类继承 BEVFusion，未重写 process_bev_features，默认只是原样返回。
        # 因此单独使用 InstanceBEVFusion 不会自动读取历史或执行时序融合。
        # 这里只保留扩展位置；实际历史处理需要相应流式实现重写该接口。
        before_decoder = getattr(self, 'fuse_before_decoder', False)
        if before_decoder:
            bev = self.process_bev_features(bev)
        #*-----------------------------------------------------------------------------#
        #* 4. 【原 DAOcc】BEV decoder 进一步编码共享特征。
        # 先执行 decoder backbone，再执行 neck；此处尚未注入检测实例信息。
        if self.decoder is not None:
            bev = self.decoder['neck'](self.decoder['backbone'](bev))
        #*-----------------------------------------------------------------------------#
        #* 5. 【近期新增：时序扩展接口】默认在 decoder 后调用。
        # 与步骤 3 二选一，每次前向只调用一次；本类默认仍为恒等操作。
        if not before_decoder:
            bev = self.process_bev_features(bev)
        #*-----------------------------------------------------------------------------#
        #* 6. 【本次新增：检测引导占据接口】进入 forward_heads，组织两个任务的交互。
        # 原 DAOcc：检测头与占据头分别读取共享 BEV，不显式传递检测实例信息。
        # 现在：ProposalHead 检测并提取候选来源网格特征 -> InstanceBEVOCCHead2D 接收实例
        # -> 内部融合模块对齐、建立空间关联、聚合消息 -> 原占据预测网络输出类别。
        # instance_to_occ 控制是否注入实例信息，具体判断位于 forward_heads。
        # 训练/推理均以预测框关联；GT 传入后仅用于训练损失，不用于实例建边。
        # forward_heads 最后按 self.training 返回损失字典或逐样本预测结果。
        return self.forward_heads(bev, metas, lidar_aug_matrix, lidar2ego,
                                  kwargs['occ_aug_matrix'], gt_bboxes_3d, gt_labels_3d, **kwargs)

    def forward_heads(self, bev, metas, lidar_aug_matrix, lidar2ego, occ_transform,
                      gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):
        """训练/推理共用候选→来源特征→占据融合；GT 只进入训练损失。"""
        # 阅读顺序：A. ProposalHead 提取候选/实例 → B. InstanceBEVOCCHead2D 准备占据特征
        # → C. BoxBEVFusion 对齐位置并建边 → D. 沿边聚合消息，再由原占据网络分类。
        # 对应可视化脚本的 decode_candidates / normal_selection / instance_to_occ_bev；
        # 脚本在预测完成后记录关系，这里在占据预测之前真正注入信息，不使用 GT 建边。
        detection, occupancy = self.heads['object'], self.heads['occ']
        #* (11.16) 1. predictions：检测头原始输出，尚未解码、筛选或做 NMS。
        # 按检测任务组织，包含 heatmap logits、中心偏移、高度、尺寸、朝向及速度等。
        # 训练时直接将 predictions 与 GT 交给 detection.loss；不是用筛选后的框计算检测损失。
        # source：检测共享卷积输出的 BEV 特征，供后续按候选来源网格提取实例向量。
        # forward_with_features 是 ProposalHead 新增接口，内部仍复用父类创建的检测网络层。
        predictions, source = detection.forward_with_features(bev, metas)
        #*--------------------------------------------------------#
        # 必须先解码候选，再计算会原地 clip_sigmoid 的检测 loss。
        # instances 是长度B的列表；每项的 boxes/features/scores/labels/source_indices 逐行对应。
        # source 是 [B,64,Xdet,Ydet] 检测特征；这里只按候选来源位置 gather，不做框内 ROI 池化。
        #* (11.16) 2. instances：供占据特征注入使用的预测实例，训练和推理均可生成。
        # 每项包含筛选后的 boxes/scores/labels/source_indices/task_ids，以及来源网格 features。
        # 框由 predictions 解码、筛选和 NMS 得到；features 从 source 提取，保留梯度。
        # 它不是 GT，也不是 detection.loss 的输入；占据损失可通过 features 回传检测特征。
        #* (11.16) 3. detections：推理最终输出的检测框、分数和类别，可用于检测评估。
        # 不含注入所需的实例特征；不用于训练检测 loss。训练分支中保持为 None。
        # 推理开启注入时，一次后处理同时产生 instances 与 detections。
        # 先保留完整检测结果，再额外筛选注入实例，注入筛选不会删减最终检测结果。
        # 旧实现保留说明：先 build_instances，末尾再 get_bboxes，会重复解码和 NMS。
        # instances = detection.build_instances(predictions, source, metas) if self.instance_to_occ else None
        detections = None
        instances = None
        if self.instance_to_occ:
            if self.training:
                #* 训练：筛选预测框仅为构建占据注入实例，不构建最终检测输出。
                instances = detection.build_instances(predictions, source, metas)
            else:
                #* 推理：共享解码与 NMS 的结果，同时返回注入实例和最终检测输出。
                instances, detections = detection.build_instances(
                    predictions, source, metas, return_detections=True)
        # 传入的 bev 仍是共享特征；占据头内部先转到 ego 网格，再融合 instances。
        # 该调用位于 training 分支之前，确保训练/推理采用同一套预测框关联规则。
        occ_pred = occupancy(bev, lidar_aug_matrix, lidar2ego, occ_transform, instances=instances)
        
        if self.training:
            #* 4. 训练监督：检测 loss 使用原始 predictions；占据 loss 使用融合后的 occ_pred。
            # 此处不使用 detections，也不以 instances 中的 NMS 框计算检测 loss。
            losses = {
                'object': detection.loss(gt_bboxes_3d, gt_labels_3d, predictions,
                                         object_targets=kwargs['object_targets']),
                'occ': occupancy.loss(occ_pred, kwargs['voxel_semantics'], kwargs['mask_camera']),
            }
            outputs = {}
            for task, values in losses.items():
                for name, value in values.items():
                    if value.requires_grad:
                        outputs[f'loss/{task}/{name}'] = value * self.loss_scale[task]
                    else:
                        outputs[f'stats/{task}/{name}'] = value
            # 关闭融合的消融实验也保持 DDP 参数图完整（新模块零梯度）。
            if not self.instance_to_occ:
                zero = sum(p.reshape(-1)[0] * 0 for p in occupancy.instance_fusion.parameters())
                outputs['loss/occ/loss_occ'] = outputs['loss/occ/loss_occ'] + zero
            return outputs
        
        #* (11.16) 5. 推理输出：训练已在上方 return，不会执行此处。
        # 开启注入时复用上方的 detections，不重复解码/NMS；关闭注入仍调用原后处理。
        # 原重复调用：detections = detection.get_bboxes(predictions, metas)
        if detections is None:
            detections = detection.get_bboxes(predictions, metas)
        occ = occupancy.get_occ(occ_pred)
        return [dict(boxes_3d=boxes.to('cpu'), scores_3d=scores.cpu(), labels_3d=labels.cpu(),
                     occ_pred=occ[b]) for b, (boxes, scores, labels) in enumerate(detections)]
