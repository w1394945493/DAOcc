"""TemporalDAOcc 历史数据加载与配对；训练时不依赖上一个 batch。"""
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
    def __init__(self, *args, temporal_mode='pair', max_time_gap=None,
                 multi_adj_frame_id_cfg=(1, 2, 1), **kwargs):
        super().__init__(*args, **kwargs)
        if temporal_mode not in ('pair', 'stream'):
            raise ValueError('temporal_mode 应为 pair 或 stream')
        self.temporal_mode, self.max_time_gap = temporal_mode, max_time_gap
        #! 与 BEVDet4D 一样，通过 range(start, stop, step) 定义历史偏移量。
        # (1,2,1) 取前1帧；(1,9,1) 取前1～8帧；(1,6,2) 取前1、3、5帧。
        if (len(multi_adj_frame_id_cfg) != 3 or
                any(type(v) is not int for v in multi_adj_frame_id_cfg)):
            raise ValueError('multi_adj_frame_id_cfg 必须为三个整数')
        start, stop, step = multi_adj_frame_id_cfg
        if start < 1 or stop <= start or step < 1:
            raise ValueError('历史偏移要求 start >= 1、stop > start、step >= 1')
        if max_time_gap is not None and max_time_gap <= 0:
            raise ValueError('max_time_gap 应为正数或 None')
        self.history_offsets = list(range(start, stop, step))
        if temporal_mode == 'stream' and self.history_offsets != [1]:
            raise ValueError('当前流式缓存仅支持前1帧；多偏移加载请使用 pair 模式')


        # 仅在已加载的 self.data_infos 中选择关键帧，不从 LiDAR sweeps 中选择。
        self.previous = {}
        scenes = {}
        for i, info in enumerate(self.data_infos):
            # 先验证场景字段，为后面的跨场景占位判断准备信息。
            if not info.get('scene_token'):
                raise ValueError('info 缺少 scene_token，请先重新生成含场景信息的 pkl')
            scenes.setdefault(info['scene_token'], []).append(i)
        self.stream_indices = []
        # 按场景、时间建立遍历顺序，但历史偏移仍相对于原 data_infos 索引。
        for scene in sorted(scenes):
            # 同场景内部按真实时间戳排序；场景之间不配对。
            indices = sorted(scenes[scene], key=lambda i: self.data_infos[i]['timestamp'])
            self.stream_indices.extend(indices)
            # 原同场景相邻索引策略不再参与选帧，保留核心代码供对照：
            # for prev, cur in zip(indices, indices[1:]):
            #     gap = (self.data_infos[cur]['timestamp']-self.data_infos[prev]['timestamp'])/1e6
            #     if 0 < gap <= max_time_gap:
            #         self.previous[cur] = prev

        #! 场景分组仅用于流式遍历；历史选择改为 BEVDet4D 的全局索引减法。
        self.history_indices = {}
        self.history_valid = {}
        for index in range(len(self.data_infos)):
            selected, valid = self.get_adj_indices(index)
            self.history_indices[index] = selected
            self.history_valid[index] = valid
            # 兼容原单历史查询接口，仅表示第一个配置偏移，不代表全部历史。
            if valid[0]:
                self.previous[index] = selected[0]

    def get_adj_indices(self, index):
        selected, valid = [], []
        info = self.data_infos[index]
        for offset in self.history_offsets:
            #! 与 BEVDet4D get_adj_info 一致：减偏移、下限截到0、检查场景。
            candidate = max(index-offset, 0)
            old = self.data_infos[candidate]
            if old['scene_token'] != info['scene_token']:
                candidate = index
            gap = (info['timestamp']-self.data_infos[candidate]['timestamp'])/1e6
            # 可选的额外时间保护；None 时不增加 BEVDet4D 以外的时间上限。
            if self.max_time_gap is not None and not 0 < gap <= self.max_time_gap:
                candidate = index
            selected.append(candidate)
            #! 保留有效性标记：占位当前帧不是实际历史，DAOcc 融合时会屏蔽它。
            valid.append(candidate != index and gap > 0)
        return selected, valid

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
            #! 用初始化时确定的映射查历史，与上个 batch 或上次可视化了哪帧无关。
            # 原单历史读取方式：
            # prev = self.previous.get(index)
            # history = self._single(prev, training) if prev is not None else None
            histories, history_meta, validity = [], [], []
            for prev, valid in zip(self.history_indices[index], self.history_valid[index]):
                history = self._single(prev, training) if valid else None
                valid = valid and history is not None
                histories.append(history if valid else dict(current))
                history_meta.append(self.get_temporal_metadata(prev) if valid else dict(meta))
                validity.append(valid)
            #! 单历史保持原有字典接口；多历史为列表，槽位顺序对应 history_offsets。
            # 固定槽位及占位数据保证不同样本能被 mmcv.collate 正确组成 batch。
            single = len(histories) == 1
            current['history'] = histories[0] if single else histories
            meta['history'] = history_meta[0] if single else history_meta
            meta['history_valid'] = validity[0] if single else validity
            meta['history_offsets'] = self.history_offsets
            meta['history_indices'] = self.history_indices[index]
        current['temporal_meta'] = DataContainer(meta, cpu_only=True)
        return current

    def prepare_train_data(self, index):
        return self._prepare(index, True)

    def prepare_test_data(self, index):
        return self._prepare(index, False)
