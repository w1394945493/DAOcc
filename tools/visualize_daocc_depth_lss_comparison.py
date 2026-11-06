#!/usr/bin/env python3
"""同帧DAOcc/DepthLSS的2×5对照，可选按场景生成视频。"""
import argparse
import gc
import json
import math
import sys
import tempfile
from pathlib import Path

from visualize_depth_lss_bev import main as run_method
from visualize_modality_comparison import save_scene_videos


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('daocc', 'depth-lss'):
        p.add_argument(f'--{name}-config', required=True)
        p.add_argument(f'--{name}-checkpoint', help='不提供则随机初始化')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--num-samples', type=int, default=20)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--scene')
    group.add_argument('--scene-index', type=int)
    p.add_argument('--num-scenes', type=int)
    p.add_argument('--all-frames', action='store_true')
    p.add_argument('--camera-mask', action='store_true')
    p.add_argument('--feature-scale', choices=('independent', 'shared'), default='independent',
                   help='independent逐图归一化；shared同帧两行的六张BEV共用百分位阈值')
    p.add_argument('--video-dir', type=Path, help='提供时，每个场景生成一个MP4')
    p.add_argument('--video-fps', type=float, default=2.)
    a = p.parse_args()
    if not math.isfinite(a.video_fps) or a.video_fps <= 0:
        p.error('video-fps必须为有限正数')
    if a.num_samples < 1 or a.workers < 0:
        p.error('num-samples必须为正数，workers不能为负数')
    return a


def draw_comparison(rows, path, scale, camera_mask):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    #* 同一帧的六张热图一起计算共享阈值；独立模式则各自取1/99百分位。
    shared = np.percentile(np.concatenate([h.ravel() for r in rows for h in r['heats']]), [1, 99])
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    try:
        for ri, row in enumerate(rows):
            name = ('DAOcc', 'DepthLSS')[ri]
            for ci, (heat, label) in enumerate(zip(row['heats'], ('Camera BEV', 'LiDAR BEV', 'Fused BEV'))):
                low, high = shared if scale == 'shared' else np.percentile(heat, [1, 99])
                norm = np.clip((heat-low)/max(float(high-low), 1e-12), 0, 1)
                axes[ri, ci].imshow(norm, origin='lower', extent=row['extent'], cmap='jet',
                                    vmin=0, vmax=1, interpolation='nearest')
                row['stats'][ci].update(color_min=float(low), color_max=float(high))
                axes[ri, ci].set_title(f'{name} — {label} (ego)' +
                                      (' (random init)' if row['random_init'] else ''), fontsize=10, pad=3)
            for ci, rgb, label in ((3, row['pred_rgb'], 'Occ prediction'), (4, row['gt_rgb'], 'Occ GT')):
                axes[ri, ci].imshow(rgb, origin='lower', extent=row['extent'], interpolation='nearest')
                suffix = ' (camera mask)' if camera_mask else ''
                if ci == 3 and row['random_init']:
                    suffix += ' (random init)'
                axes[ri, ci].set_title(f'{name} — {label} (ego){suffix}', fontsize=10, pad=3)
            for ax in axes[ri]:
                ax.set_aspect('equal')
                ax.set_axis_off()
        fig.subplots_adjust(left=.002, right=.998, bottom=.002, top=.97, wspace=.012, hspace=.07)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.02)
    finally:
        plt.close(fig)


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torchpack.utils.config import Config as TorchpackConfig
    from mmdet3d.utils import recursive_eval
    #* 两次推理必须使用相同验证数据和pipeline，避免将数据增强差异误判为方法差异。
    val_configs = []
    for path in (args.daocc_config, args.depth_lss_config):
        cfg = TorchpackConfig()
        cfg.load(str(Path(path).resolve()), recursive=True)
        val_configs.append(recursive_eval(cfg)['data']['val'])
    for key in ('type', 'ann_file', 'pipeline'):
        if val_configs[0].get(key) != val_configs[1].get(key):
            raise ValueError(f'两份配置data.val.{key}不一致，请使用相同验证集及pipeline')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, first_frames, seen = [], {}, set()
    report = dict(daocc_config=args.daocc_config, depth_lss_config=args.depth_lss_config,
                  daocc_checkpoint=args.daocc_checkpoint, depth_lss_checkpoint=args.depth_lss_checkpoint,
                  feature_scale=args.feature_scale, camera_mask=args.camera_mask,
                  coordinates='ego; horizontal X, vertical Y; origin lower', extent=[-40,40,-40,40],
                  rows=['DAOcc', 'DepthLSS'], frames=frames)
    manifest = args.output_dir / 'daocc_depth_lss_comparison_manifest.json'
    #* 先跑一个模型，再跑另一个；仅暂存CPU热图，避免两套模型同时占用GPU显存。
    # 缓存随本次运行自动清理，不长期保存中间特征或单行图片。
    with tempfile.TemporaryDirectory(prefix='.bev_compare_', dir=args.output_dir) as directory:
        cache = Path(directory)
        def collect_first(**row):
            frame = row['frame']
            token = frame['token']
            if token in first_frames:
                raise ValueError(f'重复样本token：{token}')
            path = cache / f'{token}.npz'
            np.savez_compressed(path, heats=np.stack(row['heats']), pred=row['pred_rgb'], gt=row['gt_rgb'])
            first_frames[token] = dict(frame=frame, path=path, stats=row['stats'],
                                       extent=row['extent'], random_init=row['random_init'])
            return path

        def collect_second(**row):
            token = row['frame']['token']
            if token not in first_frames or token in seen:
                raise ValueError(f'两种方法的选帧不一致或重复：{token}')
            old = first_frames[token]
            for key in ('scene_token', 'scene_frame_number'):
                if old['frame'][key] != row['frame'][key]:
                    raise ValueError(f'场景信息不一致：{token}')
            with np.load(old['path']) as saved:
                if not np.array_equal(saved['gt'], row['gt_rgb']):
                    raise ValueError(f'同帧GT可视化不一致：{token}')
                first = dict(heats=list(saved['heats']), pred_rgb=saved['pred'], gt_rgb=saved['gt'],
                             stats=old['stats'], extent=old['extent'], random_init=old['random_init'])
                frame = row['frame']
                path = args.output_dir / frame['scene'] / f"frame_{frame['scene_frame_number']:03d}_{token}.png"
                draw_comparison([first, row], path, args.feature_scale, args.camera_mask)
            frames.append(dict(**frame, path=str(path.resolve()), daocc=old['stats'], depth_lss=row['stats']))
            seen.add(token)
            manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            old['path'].unlink()
            return path

        for name, transform, callback in (('daocc', 'BEVTransformV2', collect_first),
                                           ('depth_lss', 'DepthLSSTransformDAOcc', collect_second)):
            argv = [sys.argv[0], getattr(args, name+'_config'), '--output-dir', str(cache / name)]
            for key in ('gpu', 'workers', 'num_samples', 'scene', 'scene_index', 'num_scenes', 'feature_scale'):
                value = getattr(args, key)
                if value is not None:
                    argv += ['--'+key.replace('_','-'), str(value)]
            checkpoint = getattr(args, name+'_checkpoint')
            if checkpoint:
                argv += ['--checkpoint', checkpoint]
            for key in ('all_frames', 'camera_mask'):
                if getattr(args, key):
                    argv += ['--'+key.replace('_','-')]
            previous = sys.argv
            try:
                sys.argv = argv
                run_method(expected_vtransform=transform, method_name=name, frame_callback=callback)
            finally:
                sys.argv = previous
            gc.collect()
            torch.cuda.empty_cache()
        if seen != set(first_frames):
            raise ValueError('DepthLSS未覆盖DAOcc选中的全部帧')
    if args.video_dir is not None:
        report['videos'] = save_scene_videos(frames, args.video_dir, args.video_fps)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'完成：{len(frames)}张2×5对照图；清单：{manifest}')


if __name__ == '__main__':
    main()
