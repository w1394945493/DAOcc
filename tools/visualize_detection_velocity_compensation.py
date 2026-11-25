#!/usr/bin/env python3
"""按3×N视频序列对比历史框运动补偿及速度外推。

第一行仅显示 GT 框；第二、三行以 GT 框为底层并叠加当前检测框。
第二行的历史框只做自车运动补偿，第三行的历史框先按 ``vx, vy`` 外推，
再变换到当前 LiDAR 坐标系。
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / 'configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_stream_instance_bevfusion.yaml'
DEFAULT_CHECKPOINT = Path(
    '/home/wangyushen/Data/Outputs/daocc/train/'
    'stream_pre_cbgs_faug_skip10_rev10_instance/latest.pth')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs/detection_velocity_compensation')
    parser.add_argument('--scene-index', type=int, default=0,
                        help='按 scene_token 排序后的验证场景编号')
    parser.add_argument('--scene-indices', nargs='+', type=int, default=None,
                        help='一次可视化多个验证场景编号，例如：--scene-indices 0 1 2；设置后忽略scene-index')
    parser.add_argument('--scene-token', default=None,
                        help='直接指定验证场景 token；设置后忽略 scene-index')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='所选场景内的起始帧编号')
    parser.add_argument('--num-frames', type=int, default=6,
                        help='3×N 布局中的 N，必须是连续帧')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-points', type=int, default=30000)
    parser.add_argument('--x-range', nargs=2, type=float, default=(-54., 54.))
    parser.add_argument('--y-range', nargs=2, type=float, default=(-54., 54.))
    args = parser.parse_args()
    if min(args.num_frames, args.max_points) < 1:
        parser.error('num-frames和max-points必须为正')
    if args.scene_index < 0 or args.start_frame < 0:
        parser.error('scene-index和start-frame必须非负')
    if args.scene_indices is not None and any(index < 0 for index in args.scene_indices):
        parser.error('scene-indices必须全部非负')
    if args.scene_token is not None and args.scene_indices is not None:
        parser.error('scene-token与scene-indices不能同时设置')
    if not 0 <= args.score_threshold <= 1:
        parser.error('score-threshold必须在[0,1]内')
    if args.x_range[1] <= args.x_range[0] or args.y_range[1] <= args.y_range[0]:
        parser.error('坐标范围上界必须大于下界')
    return args


def first(value):
    """解开 DataContainer/单样本列表，但保留3D框对象。"""
    from mmcv.parallel import DataContainer
    while isinstance(value, DataContainer):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        while isinstance(value, DataContainer):
            value = value.data
    return value


def homogeneous(meta, key):
    import numpy as np
    matrix = np.asarray(meta[key], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'无效的{key}: shape={matrix.shape}')
    return matrix


def previous_to_current(previous_meta, current_meta):
    """上一帧增强后 LiDAR 坐标到当前帧增强后 LiDAR 坐标。"""
    import numpy as np
    previous_global = homogeneous(previous_meta, 'ego2global') @ homogeneous(previous_meta, 'lidar2ego')
    current_global = homogeneous(current_meta, 'ego2global') @ homogeneous(current_meta, 'lidar2ego')
    previous_aug = homogeneous(previous_meta, 'lidar_aug_matrix')
    current_aug = homogeneous(current_meta, 'lidar_aug_matrix')
    #! 与 BEVFeatureWarp 的当前→历史反查矩阵方向相反：这里将历史框正向送到当前帧。
    transform = current_aug @ np.linalg.solve(current_global, previous_global) @ np.linalg.inv(previous_aug)
    if not np.isfinite(transform).all():
        raise ValueError('上一帧到当前帧的坐标变换包含非有限值')
    return transform


def propagate_boxes(boxes, previous_meta, current_meta, use_velocity):
    """将历史框变换到当前帧，并可选使用预测速度外推中心。"""
    import numpy as np
    result = boxes.copy()
    if not len(result):
        return result, float(current_meta['timestamp'] - previous_meta['timestamp'])
    if result.shape[1] < 9:
        raise ValueError('检测框不含vx、vy；期望格式[x,y,z,w,l,h,yaw,vx,vy]')
    dt = float(current_meta['timestamp'] - previous_meta['timestamp'])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f'相邻帧时间差必须为正，实际为{dt}')

    #! 第二行启用速度：速度定义在上一帧 LiDAR XY 中，先执行center += velocity * dt。
    if use_velocity:
        result[:, :2] += result[:, 7:9] * dt
    transform = previous_to_current(previous_meta, current_meta)
    centers = np.concatenate([result[:, :3], np.ones((len(result), 1))], axis=1)
    result[:, :3] = (transform @ centers.T).T[:, :3]

    #! 用中心朝向单位向量做完整坐标变换，避免直接假设两帧只有yaw旋转。
    heading = np.stack([np.cos(result[:, 6]), np.sin(result[:, 6]),
                        np.zeros(len(result)), np.zeros(len(result))], axis=1)
    heading = (transform @ heading.T).T[:, :2]
    result[:, 6] = np.arctan2(heading[:, 1], heading[:, 0])
    return result, dt


def draw_box(ax, box, color, linestyle='-', linewidth=1.0, alpha=1.0, zorder=1):
    import numpy as np
    from matplotlib.patches import Polygon
    local = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) * box[3:5] / 2
    c, s = np.cos(box[6]), np.sin(box[6])
    corners = local @ np.array([[c, s], [-s, c]]) + box[:2]
    ax.add_patch(Polygon(corners, fill=False, edgecolor=color,
                         linestyle=linestyle, linewidth=linewidth,
                         alpha=alpha, zorder=zorder))
    front = (corners[2] + corners[3]) / 2
    ax.plot([box[0], front[0]], [box[1], front[1]], color=color,
            linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)


def setup_axis(ax, points, args):
    import numpy as np
    points = points[np.isfinite(points[:, :2]).all(axis=1)]
    step = max(1, int(np.ceil(len(points) / args.max_points)))
    points = points[::step]
    ax.scatter(points[:, 0], points[:, 1], s=.12, c='.45', alpha=.28,
               linewidths=0, rasterized=True, zorder=0)
    ax.set_xlim(args.x_range); ax.set_ylim(args.y_range)
    ax.set_aspect('equal'); ax.grid(alpha=.12)
    ax.set_xlabel('LiDAR X (m)'); ax.set_ylabel('LiDAR Y (m)')


def draw_figure(path, frames, args, scene_token):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = len(frames)
    fig, axes = plt.subplots(3, columns, figsize=(5.0 * columns, 14.0), squeeze=False)
    for column, frame in enumerate(frames):
        #! 第一行只绘制 GT，便于独立观察真实目标及其跨帧运动。
        gt_ax = axes[0, column]
        setup_axis(gt_ax, frame['points'], args)
        for box in frame['gt_boxes']:
            draw_box(gt_ax, box, 'black', linewidth=.8, alpha=.9, zorder=2)
        gt_ax.set_title(
            f"GT only | frame {frame['frame_in_scene']} | GT {len(frame['gt_boxes'])}",
            fontsize=10)

        rows = [
            (axes[1, column], frame['motion_only_boxes'], 'tab:purple',
             'ego-motion only'),
            (axes[2, column], frame['motion_velocity_boxes'], 'tab:orange',
             'ego-motion + velocity'),
        ]
        for ax, historical, history_color, method in rows:
            setup_axis(ax, frame['points'], args)
            #! 后两行GT位于最底层，当前检测框居中，历史补偿框位于最上层。
            for box in frame['gt_boxes']:
                draw_box(ax, box, 'black', linewidth=.65, alpha=.65, zorder=1)
            for box in frame['boxes']:
                draw_box(ax, box, 'tab:blue', linewidth=.9, zorder=2)
            if column:
                for box in historical:
                    draw_box(ax, box, history_color, '--', linewidth=1.0,
                             alpha=.95, zorder=3)
            suffix = '' if column == 0 else f' | history {len(historical)} | dt={frame["dt"]:.3f}s'
            ax.set_title(
                f"{method} | frame {frame['frame_in_scene']} | "
                f"GT {len(frame['gt_boxes'])} | current {len(frame['boxes'])}{suffix}",
                fontsize=10)

    fig.legend(handles=[
        Line2D([], [], color='black', label='GT'),
        Line2D([], [], color='tab:blue', label='Current detections'),
        Line2D([], [], color='tab:purple', linestyle='--',
               label='Previous detections: ego-motion only'),
        Line2D([], [], color='tab:orange', linestyle='--',
               label='Previous detections: ego-motion + (vx, vy) × dt'),
    ], loc='lower center', ncol=4, frameon=True)
    fig.suptitle(f'Detection velocity compensation | scene={scene_token}', fontsize=13)
    fig.tight_layout(rect=[0, .065, 1, .95])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.apis import set_random_seed
    from torchpack.utils.config import Config as TorchpackConfig

    from mmdet3d.apis.train_stream import prepare_streaming_model_config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import recursive_eval
    from visualize_modality_comparison import clear_pretrained

    set_random_seed(0, deterministic=True)
    raw_cfg = TorchpackConfig()
    raw_cfg.load(str(args.config.resolve()), recursive=True)
    cfg = Config(recursive_eval(raw_cfg), filename=str(args.config.resolve()))
    if not cfg.get('streaming', {}).get('enabled', False):
        raise ValueError('配置必须启用streaming.enabled')
    prepare_streaming_model_config(cfg)

    #! 使用序列验证数据集，显式提供相邻帧stream_metas，模型按真实流式模式前向。
    from mmdet3d.datasets.nuscenes_sequence_dataset import NuScenesSequenceDataset
    data_cfg = deepcopy(cfg.data.val)
    data_cfg.pop('samples_per_gpu', None)
    data_cfg.update(type='NuScenesSequenceDataset', test_mode=True, sequences_split_num=1)
    dataset = build_dataset(data_cfg)
    groups = dataset.sequence_indices
    if args.scene_token is not None:
        matches = [group for group in groups
                   if dataset.data_infos[group[0]]['scene_token'] == args.scene_token]
        if not matches:
            raise ValueError(f'验证集不存在scene-token={args.scene_token}')
        selected_scenes = [(groups.index(matches[0]), matches[0])]
    else:
        scene_indices = args.scene_indices if args.scene_indices is not None else [args.scene_index]
        invalid = [index for index in scene_indices if index >= len(groups)]
        if invalid:
            raise ValueError(f'scene-indices={invalid}超出[0,{len(groups)-1}]')
        # 去重但保持命令行给定顺序。
        scene_indices = list(dict.fromkeys(scene_indices))
        selected_scenes = [(index, groups[index]) for index in scene_indices]

    torch.cuda.set_device(args.gpu)
    model_cfg = deepcopy(cfg.model)
    clear_pretrained(model_cfg)
    model = build_model(model_cfg)
    load_checkpoint(model, str(args.checkpoint.resolve()), map_location='cpu', strict=True)
    model.float().eval().cuda(args.gpu)
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in selected_scenes:
        end = args.start_frame + args.num_frames
        if end > len(scene):
            raise ValueError(
                f'场景{scene_index}只有{len(scene)}帧，无法从{args.start_frame}连续读取'
                f'{args.num_frames}帧')
        indices = scene[args.start_frame:end]
        scene_token = dataset.data_infos[indices[0]]['scene_token']

        frames = []
        previous_boxes = previous_meta = None
        #! 多场景批量绘制时逐场景清空BEV历史缓存，禁止跨场景传播历史框或特征。
        model.reset_stream()
        with torch.no_grad(), model.validation_stream():
            for frame_in_scene, index in enumerate(indices, start=args.start_frame):
                request = dict(idx=index, sequence_start=(frame_in_scene == args.start_frame),
                               slot=0, sequence_instance=scene_index,
                               sequence_direction=1)
                sample = dataset[request]
                batch = collate([sample], samples_per_gpu=1)
                data = scatter(batch, [args.gpu])[0]
                outputs = model(**data)
                output = outputs[0]

                boxes = output['boxes_3d'].tensor.detach().cpu().numpy().copy()
                scores = output['scores_3d'].detach().cpu().numpy()
                labels = output['labels_3d'].detach().cpu().numpy().astype(np.int64)
                keep = (np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
                        & (boxes[:, 3:6] > 0).all(axis=1)
                        & (scores >= args.score_threshold))
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                gt = first(data['gt_bboxes_3d'])
                gt_boxes = gt.tensor.detach().cpu().numpy().copy()
                gt_boxes[:, :3] = gt.gravity_center.detach().cpu().numpy()
                points = first(data['points']).detach().cpu().numpy()[:, :3]
                meta = first(data['stream_metas'])
                if isinstance(meta, list):
                    meta = meta[0]

                empty = np.empty((0, boxes.shape[1]), dtype=boxes.dtype)
                motion_only, motion_velocity, dt = empty, empty.copy(), 0.
                if previous_boxes is not None:
                    motion_only, dt = propagate_boxes(
                        previous_boxes, previous_meta, meta, use_velocity=False)
                    motion_velocity, velocity_dt = propagate_boxes(
                        previous_boxes, previous_meta, meta, use_velocity=True)
                    if not np.isclose(dt, velocity_dt):
                        raise RuntimeError('两种补偿方式的相邻帧时间差不一致')
                frames.append(dict(dataset_index=index, frame_in_scene=frame_in_scene,
                                   token=meta['token'], timestamp=meta['timestamp'],
                                   points=points, gt_boxes=gt_boxes, boxes=boxes,
                                   scores=scores, labels=labels,
                                   motion_only_boxes=motion_only,
                                   motion_velocity_boxes=motion_velocity, dt=dt))
                previous_boxes, previous_meta = boxes, meta

        image_path = args.output_dir / f'scene_{scene_token}_{args.start_frame:03d}_{end - 1:03d}.png'
        draw_figure(image_path, frames, args, scene_token)
        print(str(image_path.resolve()), flush=True)


if __name__ == '__main__':
    main()
