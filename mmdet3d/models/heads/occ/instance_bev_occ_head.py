"""继承原占据头，在坐标对齐后、final_conv 前注入检测实例。"""
from mmdet3d.models.builder import HEADS, build_custom
from .bev_occ_head import BEVOCCHead2D


@HEADS.register_module()
class InstanceBEVOCCHead2D(BEVOCCHead2D):
    def __init__(self, *args, instance_fusion, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_fusion = build_custom(instance_fusion)
        sampler = self.coordinate_transform
        if sampler is None or sampler.point_type != 'ego' or sampler.extra_up:
            raise ValueError('InstanceBEVOCCHead2D 要求无 extra_up 的 ego 坐标采样器')
        if self.instance_fusion.bev_channels != self.in_dim:
            raise ValueError('融合通道数必须等于 final_conv 输入通道数')

    def forward(self, img_feats, lidar_aug_matrix=None, lidar2ego=None,
                occ_aug_matrix=None, instances=None):
        # None 显式关闭实例融合，用于基线对照；空实例列表元素仍走可训练零更新。
        if instances is None:
            return super().forward(img_feats, lidar_aug_matrix, lidar2ego, occ_aug_matrix)
        if isinstance(img_feats, (list, tuple)):
            if len(img_feats) != 1:
                raise ValueError('仅支持单尺度占据 BEV')
            img_feats = img_feats[0]
        # B1. 复用原采样器：[B,512,Xdet,Ydet]先交换XY排列，再映射到ego占据平面。
        # 结果 [B,128,Yocc,Xocc] 对应可视化中的 Occupancy BEV features。
        sampler = self.coordinate_transform
        bev = sampler(img_feats.transpose(-1, -2), lidar_aug_matrix, lidar2ego, occ_aug_matrix) # (1 128 200 200)
        # B2. 对同一组ego网格查询点求其增强后LiDAR坐标，供与预测框比较。
        # points:[B,Yocc*Xocc,3] 的编号j对应 bev[b,:,j//Xocc,j%Xocc]，不能交换展平轴。
        # 变换的是位置，特征向量本身不乘 lidar2ego；不需要重采样检测图才能关联实例。
        points = self.instance_fusion.query_points(sampler, lidar_aug_matrix, lidar2ego, occ_aug_matrix) # (1 40000 3)
        bounds = (sampler.lidar_x_min, sampler.lidar_y_min, sampler.lidar_x_max, sampler.lidar_y_max)
        # B3. 用框确定连接，把实例内容注入关联单元；bounds用于剔除源BEV范围外的查询点。
        # 可视化脚本到建立边表即结束；这里进一步执行消息编码、聚合与门控残差更新。
        bev = self.instance_fusion(bev, instances, points, bounds)
        # B4. 注入位置在 coordinate_transform 之后、final_conv 之前。
        # 保留原3×3卷积和分类MLP；XY网格不变，最终由通道映射产生各高度层类别。
        prediction = self.final_conv(bev).permute(0, 3, 2, 1) # (1 200 200 128)
        batch, nx, ny = prediction.shape[:3]
        if self.use_predicter:
            prediction = self.predicter(prediction).reshape(batch, nx, ny, self.Dz, self.num_classes)
        return prediction
