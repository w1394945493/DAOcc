"""只新增派生类；通过短生命周期 hook 取得原 DAOcc Decoder 输出，不重复计算主干。"""
from mmdet3d.models.builder import FUSIONMODELS, build_head
from mmdet3d.models.fusion_models.bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class DAOccVADMap(BEVFusion):
    def __init__(self, vector_map_head, vector_map_loss_weight=1., **kwargs):
        super().__init__(**kwargs)
        if self.decoder is None or 'neck' not in self.decoder:
            raise ValueError('DAOccVADMap 要求原 DAOcc decoder.neck 输出最终 BEV')
        # 不放进原 heads 字典，避免原 forward 将矢量头误当成栅格 map 头。
        self.vector_map_head = build_head(vector_map_head)
        self.vector_map_loss_weight = vector_map_loss_weight

    def forward_single(self, img, points, camera2ego, lidar2ego, lidar2camera,
                       lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix,
                       lidar_aug_matrix, metas, gt_masks_bev=None, gt_bboxes_3d=None,
                       gt_labels_3d=None, gt_vector_map=None, **kwargs):
        features = []
        handle = self.decoder['neck'].register_forward_hook(lambda module, args, output: features.append(output))
        try:
            # 原检测/OCC前向、损失、预生成检测GT等全部复用，不复制原模型源码。
            outputs = super().forward_single(img, points, camera2ego, lidar2ego, lidar2camera,
                lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix,
                metas, gt_masks_bev, gt_bboxes_3d, gt_labels_3d, **kwargs)
        finally:
            handle.remove()
        if len(features) != 1:
            raise RuntimeError('预期 decoder.neck 仅执行一次')
        predictions = self.vector_map_head(features[0])
        if self.training:
            if gt_vector_map is None:
                raise ValueError('训练矢量头需要 VADMapNuScenesDataset 生成 gt_vector_map')
            for name, loss in self.vector_map_head.loss(predictions, gt_vector_map).items():
                # 分配/冲突数量只用于日志，不加入优化目标。
                if name.startswith('num_'):
                    outputs['stats/vector_map/'+name] = loss
                else:
                    outputs['loss/vector_map/'+name] = loss*self.vector_map_loss_weight
        else:
            for sample, vectors in zip(outputs, self.vector_map_head.decode(predictions, lidar_aug_matrix)):
                sample['vector_map'] = vectors
        return outputs
