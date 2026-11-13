"""Offline nuScenes detection-GT audit with optional fixed-frame visualization.

Defaults match the DAOcc training class list, valid_flag and symmetric BEV
range. Range counts are before random geometric augmentation. They also apply
to the current identity rotation/scale/translation + XY flip configuration.
This checks detection GT, not whether occupancy labels contain occupied voxels.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

#* (11.12) 与 vis_occ3d.py 相同的 OCC 类别配色，不导入其 Qt 环境配置。
OCC_COLORS = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0],
    [0, 150, 245], [0, 255, 255], [200, 180, 0], [255, 0, 0],
    [255, 240, 150], [135, 60, 0], [160, 32, 240], [255, 0, 255],
    [139, 137, 137], [75, 0, 75], [150, 240, 80], [230, 230, 250],
    [0, 175, 0],
], dtype=np.uint8)


def load_occ_lidar(info, root):
    from pyquaternion import Quaternion
    path = Path(info['occ3d']['occ_path']).expanduser() / 'labels.npz'
    if not path.is_file() and not path.is_absolute():
        path = root / path
    with np.load(path, allow_pickle=False) as labels:
        semantics = labels['semantics']
    if semantics.shape != (200, 200, 16):
        raise ValueError(f'Unexpected OCC shape {semantics.shape}: {path}')
    occupied = (semantics > 0) & (semantics < 17)
    indices = np.argwhere(occupied)
    ego = (indices + 0.5) * 0.4 + np.array([-40., -40., -1.])
    rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    translation = np.asarray(info['lidar2ego_translation'])
    # 行向量形式：p_lidar = (p_ego - t_lidar2ego) @ R_lidar2ego。
    lidar = (ego - translation) @ rotation
    return lidar, semantics[occupied], str(path)


#* (11.12) 固定抽取空 GT 帧，六路原始图像与当前帧 LiDAR 俯视图合并输出。
def resolve_sensor_path(value, root):
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [root / path, path]
    # 支持 pkl 中保留其他机器上的绝对路径或 data/nuscenes 前缀。
    for marker in ('samples', 'sweeps'):
        if marker in path.parts:
            candidates.append(root.joinpath(*path.parts[path.parts.index(marker):]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Sensor file not found: {value}; dataset root: {root}')


def visualize_empty_frames(infos, records, args):
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from PIL import Image

    directory = Path(args.vis_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset_root).expanduser()
    if args.vis_indices is not None:
        indices = args.vis_indices
        if any(i < 0 or i >= len(records) for i in indices):
            raise ValueError(f'--vis-indices must be in [0, {len(records) - 1}]')
        indices = list(dict.fromkeys(indices))
    else:
        indices = (np.linspace(0, len(records) - 1, min(args.vis_count, len(records)),
                               dtype=int).tolist() if records else [])
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    selected = []
    for index in indices:
        record = records[index]
        info = infos[record['source_index']]
        fig = Figure(figsize=(16, 14), dpi=120)
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 2.2])
        for k, camera in enumerate(cameras):
            path = resolve_sensor_path(info['cams'][camera]['data_path'], root)
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                rgb.thumbnail((960, 540))
                ax = fig.add_subplot(grid[k // 3, k % 3])
                ax.imshow(rgb)
            ax.set_title(camera)
            ax.axis('off')
        path = resolve_sensor_path(info['lidar_path'], root)
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 5:
            raise ValueError(f'Expected nuScenes LiDAR records with 5 float32 values: {path}')
        points = points.reshape(-1, 5)
        finite = np.isfinite(points[:, :3]).all(axis=1)
        visible = points[finite]
        lower = grid[2, :].subgridspec(1, 2, wspace=0.18)
        ax = fig.add_subplot(lower[0, 0])
        scatter = ax.scatter(visible[:, 0], visible[:, 1], c=visible[:, 2],
                             s=0.3, cmap='viridis', vmin=-3, vmax=5, linewidths=0)
        xmin, ymin, xmax, ymax = args.bev_range
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              fill=False, edgecolor='red', linewidth=1.2,
                              label='GT filtering BEV range'))
        ax.plot(0, 0, 'r+', markersize=9)
        extent = max(80., abs(xmin), abs(ymin), abs(xmax), abs(ymax))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect='equal',
               xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)',
               title=f'Current LiDAR sweep | points={len(points)} | nonfinite={int((~finite).sum())}')
        ax.legend(loc='upper right')
        # 独立色条不改变左侧绘图区尺寸，左右保持相同物理比例。
        cax = ax.inset_axes([0.03, 0.05, 0.025, 0.32])
        fig.colorbar(scatter, cax=cax, label='Z (m)')
        #* (11.12) OCC 从 ego 变换到 LiDAR 系，与点云同轴、同范围、同俯视方向。
        occ_points, occ_labels, occ_path = load_occ_lidar(info, root)
        occ_ax = fig.add_subplot(lower[0, 1], sharex=ax, sharey=ax)
        # 从低到高绘制，俯视时高处体素覆盖低处；显示完整语义 GT，不套 camera mask。
        order = np.argsort(occ_points[:, 2], kind='stable')
        occ_ax.scatter(occ_points[order, 0], occ_points[order, 1],
                       c=OCC_COLORS[occ_labels[order].astype(np.int64)] / 255.,
                       marker='s', s=1.2, linewidths=0)
        occ_ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  fill=False, edgecolor='red', linewidth=1.2))
        occ_ax.plot(0, 0, 'r+', markersize=9)
        occ_ax.set(aspect='equal', xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)',
                   title=f'OCC GT in LiDAR frame | visible voxels={len(occ_points)}\n'
                         'Classes 0/17/255 hidden; no visibility mask')
        fig.suptitle(f'Empty GT #{index} | dataset index={record["dataset_index"]} | {record["token"]}\n'
                     f'{record["reason"]} | raw={record["raw_boxes"]}, valid={record["valid_boxes"]}, '
                     f'target classes={record["recognized_boxes"]}, in range={record["in_range_boxes"]}', fontsize=12)
        fig.subplots_adjust(top=0.92, bottom=0.05, hspace=0.2, wspace=0.03)
        filename = f'{index:03d}_{record["token"]}.png'
        fig.savefig(str(directory / filename))
        fig.clear()
        selected.append(dict(record, empty_frame_index=index, image=filename,
                             occ_path=occ_path, visualization_frame='LiDAR'))
        print(f'Visualization: {directory / filename}', flush=True)
    (directory / 'selected_frames.json').write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ann_file', nargs='?', default=
                        '/home/wangyushen/Data/Datasets/nuscenes_anns/anns/daocc/nuscenes_infos_train.pkl')
    parser.add_argument('--out', default='empty_gt_report.json')
    parser.add_argument('--vis-dir', help='Optional directory for combined camera/LiDAR PNGs')
    parser.add_argument('--dataset-root', default='/home/wangyushen/Data/Datasets/Nuscenes/')
    parser.add_argument('--vis-count', type=int, default=8,
                        help='Deterministically select this many evenly spaced empty frames (default: 8)')
    parser.add_argument('--vis-indices', type=int, nargs='+',
                        help='Explicit zero-based positions in empty_frames; overrides --vis-count')
    parser.add_argument('--validity', choices=['valid_flag', 'num_lidar_pts'], default='valid_flag')
    parser.add_argument('--classes', nargs='+', default=CLASSES)
    parser.add_argument('--bev-range', nargs=4, type=float, default=[-54, -54, 54, 54],
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    args = parser.parse_args()
    if args.vis_count < 1:
        parser.error('--vis-count must be positive')
    xmin, ymin, xmax, ymax = args.bev_range
    if not xmin < xmax or not ymin < ymax:
        parser.error('Invalid BEV range')
    with open(args.ann_file, 'rb') as handle:
        infos = pickle.load(handle)['infos']
    # Preserve both source-pickle index and timestamp-sorted dataset index.
    ordered = sorted(enumerate(infos), key=lambda pair: pair[1]['timestamp'])
    counts = Counter(total_frames=len(infos))
    empty_frames, invalid_frames = [], []
    for index, (source_index, info) in enumerate(ordered):
        names = np.asarray(info['gt_names'])
        boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
        valid = (np.asarray(info['valid_flag'], dtype=bool) if args.validity == 'valid_flag'
                 else np.asarray(info['num_lidar_pts']) > 0)
        if boxes.ndim != 2 or boxes.shape[1] < 7 or len(boxes) != len(names) or valid.shape != names.shape:
            raise ValueError(f'Malformed annotation arrays for token {info["token"]}')
        recognized = valid & np.isin(names, args.classes)
        in_range = ((boxes[:, 0] > xmin) & (boxes[:, 0] < xmax)
                    & (boxes[:, 1] > ymin) & (boxes[:, 1] < ymax))
        kept = recognized & in_range
        record = dict(dataset_index=index, source_index=source_index, token=info['token'],
                      scene_token=info.get('scene_token'), timestamp=int(info['timestamp']),
                      lidar_path=info.get('lidar_path'), raw_boxes=len(names),
                      valid_boxes=int(valid.sum()), recognized_boxes=int(recognized.sum()),
                      in_range_boxes=int(kept.sum()))
        for key, condition in [('raw_empty', len(names) == 0),
                               ('empty_after_validity', not valid.any()),
                               ('empty_after_class_filter', not recognized.any()),
                               ('empty_after_range_and_class_filter', not kept.any())]:
            counts[key] += int(condition)
        if not kept.any():
            record['reason'] = ('no_raw_boxes' if not len(names) else
                                'no_valid_boxes' if not valid.any() else
                                'no_target_classes' if not recognized.any() else
                                'all_target_boxes_outside_bev')
            empty_frames.append(record)
        invalid = recognized & (~np.isfinite(boxes[:, :7]).all(axis=1)
                                | (boxes[:, 3:6] <= 0).any(axis=1))
        if invalid.any():
            invalid_frames.append(dict(record, invalid_box_indices=np.flatnonzero(invalid).tolist()))
        counts['frames_with_invalid_recognized_boxes'] += int(invalid.any())
    report = dict(ann_file=str(Path(args.ann_file).resolve()), validity=args.validity,
                  classes=args.classes, bev_range=args.bev_range,
                  scope='Detection GT only; pre-augmentation BEV range, no full pipeline execution',
                  counts=dict(counts), empty_frames=empty_frames, invalid_frames=invalid_frames)
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['counts'], indent=2))
    print(f'Empty-frame reasons: {dict(Counter(r["reason"] for r in empty_frames))}')
    print(f'Report: {output.resolve()}')
    if args.vis_dir:
        visualize_empty_frames(infos, empty_frames, args)


if __name__ == '__main__':
    main()
