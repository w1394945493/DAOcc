"""继承原 OCC 数据集，保留检测/OCC pipeline，在最终增强后追加矢量地图监督。"""
import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmdet.datasets import DATASETS
from mmdet3d.datasets.nuscenes_occupancy_dataset import NuScenesDatasetOccupancy
from .geometry import VectorMapExtractor
from .evaluation import evaluate_vectors


def unpack(value):
    while isinstance(value, (DataContainer,list,tuple)):
        if isinstance(value,DataContainer): value=value.data
        else:
            if len(value)!=1: raise ValueError('地图扩展不支持多尺度/多增强测试')
            value=value[0]
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@DATASETS.register_module()
class VADMapNuScenesDataset(NuScenesDatasetOccupancy):
    def __init__(self, vector_map=None, **kwargs):
        super().__init__(**kwargs)
        self.vector_cfg=dict(vector_map or {})
        self.vector_classes=tuple(self.vector_cfg.get('classes',['divider','ped_crossing','boundary']))
        self.extractor=VectorMapExtractor(self.dataset_root,classes=self.vector_classes,
            bounds=self.vector_cfg.get('bounds',[-30.,-15.,30.,15.]),
            num_points=self.vector_cfg.get('num_points',20))
        self._nusc_lookup=None

    def location(self,index):
        info=self.data_infos[index]
        location=info.get('map_location') or info.get('location')
        if location: return location
        # 旧 pkl 无地图区域字段时，根据 sample -> scene -> log 查找，无需重做 OCC 标注。
        if self._nusc_lookup is None:
            from nuscenes.nuscenes import NuScenes
            self._nusc_lookup=NuScenes(version=self.version,dataroot=self.dataset_root,verbose=False)
        nusc=self._nusc_lookup
        sample=nusc.get('sample',info['token'])
        scene=nusc.get('scene',sample['scene_token'])
        return nusc.get('log',scene['log_token'])['location']

    def vector_target(self,index,augmentation=None):
        return self.extractor(self.data_infos[index],self.location(index),augmentation)

    def prepare_train_data(self,index):
        example=super().prepare_train_data(index)
        if example is not None:
            augmentation=unpack(example['lidar_aug_matrix'])
            target=self.vector_target(index,augmentation)
            # 不同样本的线数量不同；cpu_only 保留每样本 dict，head 内转入 GPU。
            example['gt_vector_map']=DataContainer(target,cpu_only=True)
        return example

    def evaluate(self,results,metric='bbox',vector_map_only=False,**kwargs):
        if len(results)!=len(self): raise ValueError('预测数量与数据集不一致')
        metrics={} if vector_map_only else super().evaluate(results,metric=metric,**kwargs)
        if results and 'vector_map' in results[0]:
            # 评估 GT 使用未增强 LiDAR 坐标，预测已由 head.decode 撤销增强。
            targets=[]
            for i in range(len(self)):
                target=self.vector_target(i)
                targets.append({k:target[k] for k in ('points','labels')})
            metrics.update(evaluate_vectors([r['vector_map'] for r in results],targets,self.vector_classes))
        return metrics
