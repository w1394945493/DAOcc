"""nuScenes 矢量地图 GT：空间索引、坐标变换、裁剪和等价点序。
地图只作为监督，不作为模型推理输入。边界先从完整地图多边形生成，再裁剪，
避免把局部窗口四周人为生成成道路边界。
"""
import numpy as np
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, box
from shapely.ops import unary_union, linemerge
from shapely.strtree import STRtree


def lines(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type in ('LineString', 'LinearRing'):
        return [LineString(geometry.coords)] if geometry.length > 1e-4 else []
    return [line for part in getattr(geometry, 'geoms', []) for line in lines(part)]


def sample_line(line, count):
    return np.asarray([line.interpolate(t).coords[0][:2] for t in
                       np.linspace(0, line.length, count)], dtype=np.float32)


def equivalent_orders(line, count):
    """开曲线允许正反序；闭曲线还允许改变起点，所有合法表示参与匹配。"""
    pts = sample_line(line, count)
    variants = [pts, pts[::-1].copy()]
    if line.is_ring:
        base = pts[:-1]
        variants = []
        for reverse in (base, base[::-1]):
            for shift in range(len(base)):
                rolled = np.roll(reverse, shift, axis=0)
                variants.append(np.concatenate([rolled, rolled[:1]], 0))
    return pts, variants


def planar_lidar_to_global(info):
    # 与 VAD 地图提取一致：用 LiDAR 的全局位置及 yaw 定义局部二维地图。
    ego = Quaternion(info['ego2global_rotation'])
    lidar = Quaternion(info['lidar2ego_rotation'])
    rotation = (ego*lidar).rotation_matrix
    translation = ego.rotate(info['lidar2ego_translation'])+np.asarray(info['ego2global_translation'])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,translation[0]], [s,c,translation[1]], [0,0,1.]])


def planar_augmentation(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all():
        raise ValueError('lidar_aug_matrix 必须是有限的4x4矩阵')
    if not np.allclose(matrix[:2, 2], 0, atol=1e-5) or not np.allclose(matrix[2, :2], 0, atol=1e-5):
        raise ValueError('地图监督当前支持平面旋转/缩放/翻转/平移，不支持非平面BDA')
    out = np.eye(3)
    out[:2,:2], out[:2,2] = matrix[:2,:2], matrix[:2,3]
    if abs(np.linalg.det(out)) < 1e-8:
        raise ValueError('增强矩阵不可逆')
    return out


def transform_geometry(geometry, matrix):
    return affinity.affine_transform(geometry, [matrix[0,0],matrix[0,1],matrix[1,0],
                                               matrix[1,1],matrix[0,2],matrix[1,2]])


class VectorMapExtractor:
    def __init__(self, root, classes=('divider','ped_crossing','boundary'),
                 bounds=(-30.,-15.,30.,15.), num_points=20):
        self.root, self.classes, self.bounds, self.num_points = root, tuple(classes), bounds, num_points
        if not set(classes) <= {'divider','ped_crossing','boundary'}:
            raise ValueError('支持 divider/ped_crossing/boundary')
        self._locations = {}

    def _load(self, location):
        if location not in self._locations:
            from nuscenes.map_expansion.map_api import NuScenesMap
            api = NuScenesMap(dataroot=self.root, map_name=location)
            index = {}
            required = []
            if 'divider' in self.classes: required += ['road_divider','lane_divider']
            if 'ped_crossing' in self.classes: required += ['ped_crossing']
            if 'boundary' in self.classes: required += ['road_segment','lane']
            for layer in required:
                geometries = []
                for record in getattr(api, layer):
                    if layer.endswith('divider'):
                        geom = api.extract_line(record['line_token'])
                    else:
                        geom = api.extract_polygon(record['polygon_token'])
                        if not geom.is_valid: geom = geom.buffer(0)
                    if not geom.is_empty: geometries.append(geom)
                index[layer] = (geometries, STRtree(geometries) if geometries else None)
            self._locations[location] = index
        return self._locations[location]

    def __call__(self, info, location, augmentation=None):
        augmentation = np.eye(4) if augmentation is None else augmentation
        # global -> 未增强 LiDAR -> 增强后的 LiDAR，GT 与最终 BEV 使用同一坐标。
        transform = planar_augmentation(augmentation) @ np.linalg.inv(planar_lidar_to_global(info))
        roi = box(*self.bounds)
        global_roi = transform_geometry(roi, np.linalg.inv(transform))
        index = self._load(location)
        def query(layer):
            geoms, tree = index[layer]
            if tree is None: return []
            # Shapely 1.x 返回几何对象，2.x 返回索引；同时兼容两者。
            found = tree.query(global_roi)
            found = [geoms[int(g)] if isinstance(g, (int,np.integer)) else g for g in found]
            return [g for g in found if g.intersects(global_roi)]
        all_lines, labels = [], []
        for label, name in enumerate(self.classes):
            if name == 'divider':
                sources = query('road_divider')+query('lane_divider')
            else:
                polygons = query('ped_crossing') if name == 'ped_crossing' else query('road_segment')+query('lane')
                sources = lines(unary_union(polygons).boundary) if polygons else []
            for source in sources:
                clipped = transform_geometry(source, transform).intersection(roi)
                if clipped.geom_type == 'MultiLineString':
                    clipped = linemerge(clipped)
                for line in lines(clipped):
                    all_lines.append(line); labels.append(label)
        count, p = len(all_lines), self.num_points
        orders = np.zeros((count, max(2,2*(p-1)),p,2), dtype=np.float32)
        mask = np.zeros(orders.shape[:2], dtype=bool)
        points = np.zeros((count,p,2), dtype=np.float32)
        for i,line in enumerate(all_lines):
            points[i], variants = equivalent_orders(line,p)
            orders[i,:len(variants)] = variants
            mask[i,:len(variants)] = True
        return dict(points=points, labels=np.asarray(labels,dtype=np.int64),orders=orders,order_mask=mask)
