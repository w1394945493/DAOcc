"""保留原DepthLSS，实现适用于DAOcc双模态分支的独立适配版本。"""

from mmdet3d.models.builder import VTRANSFORMS
from .depth_lss import DepthLSSTransform

__all__ = ["DepthLSSTransformDAOcc"]


@VTRANSFORMS.register_module()
class DepthLSSTransformDAOcc(DepthLSSTransform):
    """LiDAR深度引导LSS，输出与DAOcc一致的[B, Z*C, X, Y]特征。

    使用父类的深度编码、深度概率预测和BEV pooling；不改变原类行为。
    此版本仅支持downsample=1，避免父类下采样层按C而非Z*C配置通道。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #* 多高度层拼接后为Z*C通道，原DepthLSS的下采样卷积仅接收C通道。
        # 当前实验保留180×180空间尺寸；需要下采样时应单独设计BEV适配层。
        from torch import nn
        if not isinstance(self.downsample, nn.Identity):
            raise ValueError("DepthLSSTransformDAOcc要求downsample=1")

    def forward(
        self, img, points, camera2ego, lidar2ego, lidar2camera,
        lidar2image, camera_intrinsics, camera2lidar,
        img_aug_matrix, lidar_aug_matrix, metas, **kwargs,
    ):
        #* 1. 保护原始点云；支持B个样本组成的list，每项为[Ni, 点特征维数]。
        # 父类在撤销三维增强时会对XYZ切片原地减去平移量。
        # clone保留梯度连接，但隔离存储，避免后续LiDAR体素化读到被修改的点。
        if points is None or len(points) != img.shape[0]:
            raise ValueError("DepthLSS需要与图像batch大小相同的LiDAR点云列表")
        protected_points = [p.clone() for p in points]
        if tuple(img.shape[-2:]) != tuple(self.feature_size):
            raise ValueError(
                f"图像特征尺寸{tuple(img.shape[-2:])}与feature_size "
                f"{tuple(self.feature_size)}不一致"
            )

        #* 2. 父类：点云投影为稀疏深度图，与图像特征一起预测深度分布。
        # 深度加权图像特征经三维定位和BEV pooling，沿高度拼接通道。
        # 输入img：[B,Ncam,Cin,Hf,Wf]；返回bev：[B,Z*C,X,Y]。
        # 父类的force_fp32装饰器继续负责混合精度下的几何/池化计算。
        bev = super().forward(
            img, protected_points, camera2ego, lidar2ego, lidar2camera,
            lidar2image, camera_intrinsics, camera2lidar,
            img_aug_matrix, lidar_aug_matrix, metas, **kwargs,
        )
        
        #* 3. 保持父类的[X,Y]排列，使融合前两分支索引对应同一物理轴。
        # 原实现误将DAOcc约定判断为[Y,X]，下面保留旧注释和语句供追溯，禁止执行。
        # 【旧注释，轴顺序判断有误】将空间轴转换为DAOcc原BEVTransformV2采用的[Y,X]顺序。
        # 【旧注释】只交换空间轴，不翻转物理坐标，不改变通道内的高度顺序。
        # 【旧注释】本配置：[B,128*10,180,180]；即使X/Y尺寸相等，也不能省略转置。
        # return bev.transpose(-1, -2).contiguous()

        #* BEV pooling输出[B,C,Z,X,Y]，高度拼接后已经是[B,Z*C,X,Y]。
        # 当前SparseEncoderSpconv2和原BEVTransformV2也输出[X,Y]，无需再次转置。
        # 例如物理索引(X=130,Y=70)应保留在[...,130,70]；转置会错误移到[...,70,130]。
        # 两轴都为180时shape不变，但物理位置会错位；contiguous仅整理存储，不交换轴。
        # Occ head内部的空间轴交换及LiDAR→ego采样仍按原流程执行。
        return bev.contiguous()
