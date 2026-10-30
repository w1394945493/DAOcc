"""spconv 2.x PointToVoxel 的 DAOcc 兼容包装器。"""

import torch
from torch import nn

from spconv.pytorch.utils import PointToVoxel


class Spconv2Voxelization(nn.Module):
    """使用 spconv2 生成硬体素，并保持原 ``Voxelization`` 输出接口。

    输入参数仍使用 DAOcc 配置中的 XYZ 顺序。spconv2 返回的坐标为
    ZYX，本类将其转回 DAOcc 现有代码使用的 XYZ，因此旧、新
    ``SparseEncoder`` 都可以接收这里的输出。

    ``PointToVoxel`` 必须在 CUDA device 已确定后创建，所以生成器在
    第一次 forward 时按 device、点特征数和 max_voxels 延迟构建。
    """

    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels=20000,
        num_point_features=None,
    ):
        super().__init__()
        if len(voxel_size) != 3:
            raise ValueError(f"voxel_size 必须为 [x,y,z]，实际为: {voxel_size}")
        if len(point_cloud_range) != 6:
            raise ValueError(
                "point_cloud_range 必须为 [xmin,ymin,zmin,xmax,ymax,zmax]，"
                f"实际为: {point_cloud_range}"
            )
        if max_num_points <= 0:
            raise ValueError("Spconv2Voxelization 仅实现硬体素化，max_num_points 必须大于 0")

        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.point_cloud_range = tuple(float(v) for v in point_cloud_range)
        self.max_num_points = int(max_num_points)
        self.max_voxels = self._normalize_max_voxels(max_voxels)
        self.num_point_features = (
            None if num_point_features is None else int(num_point_features)
        )

        point_cloud_range_tensor = torch.tensor(
            self.point_cloud_range, dtype=torch.float32
        )
        voxel_size_tensor = torch.tensor(self.voxel_size, dtype=torch.float32)
        self.grid_size = torch.round(
            (point_cloud_range_tensor[3:] - point_cloud_range_tensor[:3])
            / voxel_size_tensor
        ).long()
        self.pcd_shape = [*self.grid_size[:2], 1]

        # PointToVoxel 不是 nn.Module，且持有指定 device 上的预分配张量；
        # 不把它注册为子模块，按实际输入设备延迟创建并缓存。
        self._voxel_generators = {}

    @staticmethod
    def _normalize_max_voxels(max_voxels):
        if isinstance(max_voxels, int):
            return (max_voxels, max_voxels)
        values = tuple(int(v) for v in max_voxels)
        if len(values) != 2:
            raise ValueError(
                "max_voxels 必须是整数或 [train, test] 两项，"
                f"实际为: {max_voxels}"
            )
        return values

    def _get_generator(self, points, max_voxels):
        if points.dtype != torch.float32:
            raise TypeError(
                "spconv2 PointToVoxel 要求 FP32 点云；BEVFusion.voxelize() "
                f"应通过 force_fp32 转换，实际 dtype={points.dtype}"
            )
        num_point_features = int(points.shape[1])
        if (
            self.num_point_features is not None
            and num_point_features != self.num_point_features
        ):
            raise ValueError(
                f"点特征数应为 {self.num_point_features}，实际为 {num_point_features}"
            )

        device_index = points.device.index if points.device.index is not None else -1
        key = (points.device.type, device_index, num_point_features, max_voxels)
        generator = self._voxel_generators.get(key)
        if generator is None:
            generator = PointToVoxel(
                vsize_xyz=list(self.voxel_size),
                coors_range_xyz=list(self.point_cloud_range),
                num_point_features=num_point_features,
                max_num_voxels=max_voxels,
                max_num_points_per_voxel=self.max_num_points,
                device=points.device,
            )
            self._voxel_generators[key] = generator
        return generator

    @torch.no_grad()
    def forward(self, points):
        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]
        generator = self._get_generator(points, max_voxels)
        voxels, coordinates_zyx, num_points = generator(points)

        # spconv2 [z,y,x] -> DAOcc 既有接口 [x,y,z]。BEVFusion 随后只需
        # 像原实现一样在最前面补 batch 索引，不影响任何下游模块。
        coordinates_xyz = coordinates_zyx[:, [2, 1, 0]].contiguous()
        return voxels, coordinates_xyz, num_points

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(voxel_size={self.voxel_size}, "
            f"point_cloud_range={self.point_cloud_range}, "
            f"max_num_points={self.max_num_points}, "
            f"max_voxels={self.max_voxels}, "
            f"num_point_features={self.num_point_features})"
        )

