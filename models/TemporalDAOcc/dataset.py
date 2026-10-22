"""显式构造同场景相邻帧，训练时不依赖上一个 batch。"""
import numpy as np
from pyquaternion import Quaternion
from mmcv.parallel import DataContainer
from mmdet.datasets import DATASETS
from mmdet3d.datasets.nuscenes_occupancy_dataset import NuScenesDatasetOccupancy


def pose(info):
    transforms = []
    for prefix in ('ego2global', 'lidar2ego'):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = Quaternion(info[prefix+'_rotation']).rotation_matrix
        matrix[:3, 3] = info[prefix+'_translation']
        transforms.append(matrix)
    return transforms[0] @ transforms[1]


@DATASETS.register_module()
class TemporalNuScenesDataset(NuScenesDatasetOccupancy):
    def __init__(self, *args, temporal_mode='pair', max_time_gap=1., **kwargs):
        super().__init__(*args, **kwargs)
        if temporal_mode not in ('pair', 'stream'):
            raise ValueError('temporal_mode 应为 pair 或 stream')
        self.temporal_mode, self.max_time_gap = temporal_mode, max_time_gap
        self.previous = {}
        scenes = {}
        for i, info in enumerate(self.data_infos):
            if not info.get('scene_token'):
                raise ValueError('info 缺少 scene_token，请先重新生成含场景信息的 pkl')
            scenes.setdefault(info['scene_token'], []).append(i)
        self.stream_indices = []
        for scene in sorted(scenes):
            indices = sorted(scenes[scene], key=lambda i: self.data_infos[i]['timestamp'])
            self.stream_indices.extend(indices)
            for prev, cur in zip(indices, indices[1:]):
                gap = (self.data_infos[cur]['timestamp']-self.data_infos[prev]['timestamp'])/1e6
                if 0 < gap <= max_time_gap:
                    self.previous[cur] = prev

    # * 父类 self.metadata 是 pkl 中的元信息字典，不能用同名方法，否则被属性遮蔽。
    def get_temporal_metadata(self, index):
        info = self.data_infos[index]
        return dict(scene=info['scene_token'], token=info['token'],
                    timestamp=float(info['timestamp'])/1e6, lidar2global=pose(info))

    def _single(self, index, training):
        # 调用父类单帧 pipeline，保留原有多 sweep 点云与增强流程。
        method = super().prepare_train_data if training else super().prepare_test_data
        return method(index)

    def _prepare(self, index, training):
        current = self._single(index, training)
        if current is None:
            return None
        meta = self.get_temporal_metadata(index)
        if self.temporal_mode == 'pair':
            prev = self.previous.get(index)
            history = self._single(prev, training) if prev is not None else None
            valid = history is not None
            # 占位历史保持 batch 可堆叠；模型会屏蔽无效历史。
            current['history'] = history if valid else dict(current)
            meta['history'] = self.get_temporal_metadata(prev) if valid else dict(meta)
            meta['history_valid'] = valid
        current['temporal_meta'] = DataContainer(meta, cpu_only=True)
        return current

    def prepare_train_data(self, index):
        return self._prepare(index, True)

    def prepare_test_data(self, index):
        return self._prepare(index, False)
