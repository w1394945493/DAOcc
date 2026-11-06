"""可选 DAOcc 矢量地图扩展。显式导入本包才注册，不改原有 __init__.py。"""
from .head import VADVectorMapHead
from .model import DAOccVADMap
from .dataset import VADMapNuScenesDataset

__all__ = ['VADVectorMapHead', 'DAOccVADMap', 'VADMapNuScenesDataset']
