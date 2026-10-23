"""保持BEVFusion调用接口，替换最终占据解码方式。"""
import torch
from mmdet3d.models.builder import HEADS
from mmdet3d.models.heads.occ.bev_occ_head import BEVOCCHead2D
from .geometry import GaussianPredictor
from .aggregation import GaussianAggregation


@HEADS.register_module()
class GaussianOccHead(BEVOCCHead2D):
    def __init__(self, gaussian=None, **kwargs):
        super().__init__(**kwargs)
        if self.num_classes != 18 or not self.use_predicter:
            raise ValueError('当前高斯头要求 num_classes=18，use_predicter=true')
        #* 保留父类坐标变换、卷积和损失；移除实例中的旧MLP，原源码不作修改。
        del self.predicter
        config = dict(gaussian or {})
        bounds = config.pop('bounds', [-40., -40., -1., 40., 40., 5.4])
        shape = config.pop('grid_shape', [200, 200, 16])
        if shape[2] != self.Dz:
            raise ValueError('grid_shape的Z维必须与Dz一致')
        backend = config.pop('backend', 'cuda')
        multiplier = config.pop('scale_multiplier', 3.)
        if kwargs.get('coordinate_transform'):
            transform = kwargs['coordinate_transform']
            # 坐标变换参数的顺序是Y范围、X范围；高斯边界是XYZ下界、XYZ上界。
            expected = [bounds[1], bounds[4], bounds[0], bounds[3]]
            if list(transform['point_range'][:4]) != expected:
                raise ValueError('coordinate_transform的XY范围必须与高斯范围一致')
        self.gaussian_predictor = GaussianPredictor(self.out_dim, bounds, **config)
        self.gaussian_aggregation = GaussianAggregation(bounds, shape, multiplier, backend)

    def forward(self, img_feats, lidar_aug_matrix=None, lidar2ego=None, occ_aug_matrix=None):
        if isinstance(img_feats, (list, tuple)):
            if len(img_feats) != 1:
                raise ValueError('高斯头期望单尺度BEV特征')
            img_feats = img_feats[0]
        #* 沿用原头的XY转换与ego/增强坐标对齐，输出高斯直接处于占据标签坐标系。
        feature = img_feats.transpose(-1, -2) # (1 512 180 180)
        if self.coordinate_transform is not None:
            feature = self.coordinate_transform(feature, lidar_aug_matrix, lidar2ego, occ_aug_matrix)
        feature = self.final_conv(feature) # (1 128 200 200)
        return self.gaussian_aggregation(self.gaussian_predictor(feature))

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        #* occ_pred是归一化概率的log，CE(log(p),y)等于NLL；不能直接把p交给CE。
        labels = voxel_semantics.long().reshape(-1)
        valid = (labels >= 0) & (labels < self.num_classes)
        if self.use_mask:
            valid = valid & mask_camera.reshape(-1).bool()
        if not valid.any():
            return dict(loss_occ=occ_pred.sum()*0.)
        labels = labels[valid]
        predictions = occ_pred.reshape(-1, self.num_classes)[valid]
        if self.class_balance:
            denominator = self.cls_weights.to(predictions.device)[labels].sum().clamp(min=1e-8)
        else:
            denominator = labels.numel()
        return dict(loss_occ=self.loss_occ(predictions, labels, avg_factor=denominator))
