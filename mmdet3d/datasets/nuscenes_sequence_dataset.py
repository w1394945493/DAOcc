"""Ordered nuScenes training samples; never silently replace a rejected frame."""
import math
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from mmcv.parallel import DataContainer
from mmdet.datasets import DATASETS
from .nuscenes_occupancy_dataset import NuScenesDatasetOccupancy
from .pipelines.transforms_3d import ImageAug3D, GlobalRotScaleTrans, RandomFlip3D


@DATASETS.register_module()
class NuScenesSequenceDataset(NuScenesDatasetOccupancy):
    def __init__(self, *args, sequences_split_num=1, keep_consistent_seq_aug=True, **kwargs):
        #* (11.12) 沿用 StreamOcc 的参数：1 不切分，整数 K 按每场景 K 段切分，"all" 每帧一组。
        # 默认值与 StreamOcc Dataset 一致；不引入按固定帧数切分的另一套参数。
        self.sequences_split_num = sequences_split_num
        #* (11.12) 与 StreamOcc 同名同义：True 为子序列内复用，False 为逐帧采样。
        if not isinstance(keep_consistent_seq_aug, (bool, np.bool_)):
            raise TypeError('keep_consistent_seq_aug must be a boolean')
        self.keep_consistent_seq_aug = bool(keep_consistent_seq_aug)
        kwargs.update(filter_empty_gt=False, resample=False)
        super().__init__(*args, **kwargs)
        self._stream_aug_transforms = []
        seen = set()
        for transform in self.pipeline.transforms:
            for kind in (ImageAug3D, GlobalRotScaleTrans, RandomFlip3D):
                if isinstance(transform, kind):
                    if kind in seen:
                        raise ValueError(f'Duplicate {kind.__name__} in sequence pipeline')
                    seen.add(kind)
                    self._stream_aug_transforms.append(transform)
                    break
        scenes = {}
        for index, info in enumerate(self.data_infos):
            scene = info.get('scene_token')
            if not scene:
                raise ValueError('Sequence training requires scene_token in every info')
            scenes.setdefault(scene, []).append(index)
        self.sequence_indices = []
        self.flag = np.empty(len(self.data_infos), dtype=np.int64)
        for group, scene in enumerate(sorted(scenes)):
            indices = sorted(scenes[scene], key=lambda i: self.data_infos[i]['timestamp'])
            times = [self.data_infos[i]['timestamp'] for i in indices]
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError(f'Non-increasing timestamps in scene {scene}')
            self.sequence_indices.append(indices)
            self.flag[indices] = group

        self._split_sequence_groups()

    def get_augmentation(self, index=0, rng=None):
        """Sample explicit 2D/3D geometry using existing DAOcc pipeline ranges.

        Called in the sampler process. Parameters travel with each request,
        so worker assignment/prefetch does not affect sequence consistency.
        Like StreamOcc aug_config, this does not freeze GridMask/point shuffle.
        """
        if self.test_mode:
            return None
        aug_config = {'2d': [], '3d': {}}
        for transform in self._stream_aug_transforms:
            if isinstance(transform, ImageAug3D):
                cameras = self.data_infos[index].get('cams', {})
                if not cameras:
                    raise ValueError('Image augmentation requires nuScenes camera metadata')
                # 每相机可不同，但同一相机跨帧复用，保持 DAOcc 的相机增强分布。
                aug_config['2d'] = [transform.get_augmentation(rng) for _ in cameras]
            else:
                aug_config['3d'].update(transform.get_augmentation(rng))
        return aug_config

    def _split_sequence_groups(self):
        #* (11.12) 参照 StreamOcc Dataset._set_sequence_group_flag 的第二阶段切分。
        # DAOcc 仍按 scene_token 建立原始序列；对排序后的实际帧索引切片，再回填 flag，
        # 避免假定不同 scene 在 data_infos 中必须连续。切分步长与组数校验沿用 StreamOcc。
        if self.sequences_split_num == 1:
            return
        if self.sequences_split_num == "all":
            self.flag = np.arange(len(self.data_infos), dtype=np.int64)
            self.sequence_indices = [[i] for i in range(len(self.data_infos))]
            return

        original_groups = self.sequence_indices
        split_groups = []
        for indices in original_groups:
            length = len(indices)
            boundaries = list(range(
                0, length, math.ceil(length / self.sequences_split_num)
            )) + [length]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                split_groups.append(indices[start:end])

        # 与 StreamOcc 一样，过短序列等导致实际组数不符合 K 倍时失败，不自动补帧或降级。
        assert sum(len(group) for group in split_groups) == len(self.flag)
        assert len(split_groups) == len(original_groups) * self.sequences_split_num
        self.sequence_indices = split_groups
        for group_id, indices in enumerate(split_groups):
            self.flag[indices] = group_id

    def __getitem__(self, request):
        #* (11.12) 训练中流式验证接收验证采样器的请求，使用原验证 pipeline。
        # 普通整数索引的验证访问仍沿用父类，原单帧数据接口不变。
        if self.test_mode and not isinstance(request, dict):
            return super().__getitem__(request)
        if not isinstance(request, dict):
            raise TypeError('NuScenesSequenceDataset requires a sequence sampler request')
        index = request['idx']
        aug_config = None
        if self.test_mode:
            data = self.prepare_test_data(index)
        else:
            #* (11.12) 参照 StreamOcc 将采样器增强参数送入 pipeline，不依赖 worker 内状态。
            # 本数据集强制 filter_empty_gt=False，保留原 prepare_train_data 的读取/预处理顺序。
            if request.get('aug_config') is None:
                raise ValueError('Sequence training requests must include aug_config')
            aug_config = deepcopy(request['aug_config'])
            input_dict = self.get_data_info(index)
            self.pre_pipeline(input_dict)
            input_dict['aug_config'] = deepcopy(aug_config)
            data = self.pipeline(input_dict)
        if data is None:
            raise RuntimeError(f'Pipeline rejected sequence frame {index}; random replacement is forbidden')
        info = self.data_infos[index]
        #* (11.12) 保存当前帧真实使用的 LiDAR 外参和最终增强矩阵（包括随机翻转）。
        # global 位姿由原始标注以 FP64 构造，避免大坐标先转 FP32 后相减损失精度。
        # 使用普通嵌套 list，便于 CPU DataContainer、缓存元数据比较及 JSON 记录。
        ego2global = np.eye(4, dtype=np.float64)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        if not np.isfinite(ego2global).all():
            raise ValueError(f'Invalid ego2global in sequence frame {index}')
        geometry = dict(ego2global=ego2global.tolist())
        for key in ('lidar2ego', 'lidar_aug_matrix'):
            if key not in data:
                raise KeyError(f'Sequence pipeline must collect {key} for BEV alignment')
            value = data[key]
            if isinstance(value, DataContainer):
                value = value.data
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float64)
            if (value.shape != (4, 4) or not np.isfinite(value).all()
                    or not np.allclose(value[3], [0, 0, 0, 1])):
                raise ValueError(f'Invalid {key} in sequence frame {index}')
            geometry[key] = value.tolist()
        # 记录实际图像/占据增强矩阵，供跨帧一致性及模型输入核对使用。
        for key in ('img_aug_matrix', 'occ_aug_matrix'):
            if key in data:
                value = data[key]
                if isinstance(value, DataContainer):
                    value = value.data
                if hasattr(value, 'detach'):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value, dtype=np.float64)
                if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {key} in sequence frame {index}')
                geometry[key] = value.tolist()
        # Added after Collect3D, so the original pipeline and its output keys stay intact.
        data['stream_metas'] = DataContainer(dict(
            dataset_index=index, scene_token=info['scene_token'], token=info['token'],
            timestamp=float(info['timestamp']) / 1e6,
            sequence_start=request['sequence_start'], slot=request['slot'],
            sequence_instance=request['sequence_instance'],
            aug_config=aug_config,
            augmentation_mode=('validation' if self.test_mode else
                               'sequence' if self.keep_consistent_seq_aug else 'frame'),
            **geometry,
        ), cpu_only=True)
        return data
