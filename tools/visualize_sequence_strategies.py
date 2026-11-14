"""Visualize streaming sampling and actual augmented inputs without a model.

Each selected original group is compared under five controlled variants. The
separate sampling overview uses the actual global sampler with configured CBGS.
Output is placed in a new run directory; existing results are never deleted.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_training_sampling import load_config


def save_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert) + '\n')


def unwrap(value):
    from mmcv.parallel import DataContainer
    return value.data if isinstance(value, DataContainer) else value


def timeline(ax, indices, requests, events):
    positions = {idx: p for p, idx in enumerate(indices)}
    ax.scatter(range(len(indices)), [0] * len(indices), c='lightgray', label='original')
    for reason, color, marker in [('random_skip', 'orange', 'x'), ('empty_gt', 'red', 'x')]:
        xs = [positions[e['idx']] for e in events if e['reason'] == reason]
        ax.scatter(xs, [0] * len(xs), c=color, marker=marker, s=70, label=reason)
    xs = [positions[r['idx']] for r in requests]
    ax.scatter(xs, [0] * len(xs), c='green', label='emitted')
    for order, x in enumerate(xs):
        ax.text(x, .05, str(order), ha='center', fontsize=7)
    for a, b in zip(xs, xs[1:]):
        ax.annotate('', xy=(b, -.1), xytext=(a, -.1), arrowprops=dict(arrowstyle='->', color='blue'))
    ax.set(xlim=(-1, len(indices)), ylim=(-.3, .3), yticks=[], xlabel='Original position; labels = reading step')
    ax.legend(loc='upper left', fontsize=7, ncol=4)


def draw_frame(dataset, request, path, title, indices, requests, events, mean, std, limit, max_points):
    import matplotlib.pyplot as plt
    # Same frame has identical worker-side randomness across comparison variants.
    seed = (request['idx'] + 1009) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    data = dataset[deepcopy(request)]
    img = unwrap(data['img'])
    if isinstance(img, (list, tuple)):
        img = torch.stack(img)
    images = img.detach().cpu().numpy().transpose(0, 2, 3, 1)
    images = np.clip(images * std + mean, 0, 1)
    cameras = list(dataset.data_infos[request['idx']]['cams'])
    fig = plt.figure(figsize=(17, 12))
    grid = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.8, .45])
    preferred = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for k, camera in enumerate(preferred):
        ax = fig.add_subplot(grid[k // 3, k % 3])
        ax.imshow(images[cameras.index(camera)])
        ax.set_title(camera)
        ax.axis('off')
    ax = fig.add_subplot(grid[2, :2])
    points = unwrap(data['points']).detach().cpu().numpy()
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    if len(points) > max_points:
        points = points[np.random.RandomState(0).choice(len(points), max_points, replace=False)]
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=.2, cmap='viridis', vmin=-3, vmax=5)
    boxes = unwrap(data['gt_bboxes_3d']).corners.detach().cpu().numpy()
    labels = unwrap(data['gt_labels_3d']).detach().cpu().numpy()
    for box, label in zip(boxes, labels):
        polygon = box[[0, 3, 7, 4, 0], :2]
        ax.plot(polygon[:, 0], polygon[:, 1], color=plt.get_cmap('tab10')(int(label) % 10), linewidth=.7)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='Augmented LiDAR X (m)', ylabel='Y (m)', title='Pipeline point cloud (including sweeps) + filtered GT')
    info = fig.add_subplot(grid[2, 2]); info.axis('off')
    meta = unwrap(data['stream_metas'])
    aug = request['aug_config']
    text = 'Geometry augmentation (full metadata in JSON)\n3D:\n'
    text += '\n'.join('%s: %s' % (k, v) for k, v in aug.get('3d', {}).items())
    for camera, params in zip(cameras, aug.get('2d', [])):
        text += '\n\n' + camera + '\n' + ', '.join(
            '%s=%s' % (k, round(v, 3) if isinstance(v, float) else v) for k, v in params.items())
    info.text(0, 1, text, va='top', fontsize=7, wrap=True)
    timeline(fig.add_subplot(grid[3, :]), indices, requests, events)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--num-groups', type=int, default=2)
    parser.add_argument('--group-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-prob', type=float, default=.3)
    parser.add_argument('--flip-prob', type=float, default=1., help='Default 1 demonstrates reversal; may set a probability.')
    parser.add_argument('--slots', type=int, default=4, help='Global slots in scheduling overview (CPU simulation).')
    parser.add_argument('--steps', type=int, default=80, help='Scheduling overview steps, not per-group rendering limit.')
    parser.add_argument('--max-frames', type=int, default=0, help='Render first N emitted frames per variant; 0 renders all. Sampling trace always complete.')
    parser.add_argument('--max-points', type=int, default=80000)
    parser.add_argument('--lidar-limit', type=float, default=54.)
    parser.add_argument('--trace-only', action='store_true', help='Render sampling timelines only, without sensor loading.')
    args = parser.parse_args()
    if min(args.num_groups, args.slots, args.steps, args.max_points) < 1 or args.max_frames < 0 or args.lidar_limit <= 0:
        parser.error('Counts/limits must be positive (max-frames can be zero)')
    if not 0 <= args.skip_prob <= 1 or not 0 <= args.flip_prob <= 1:
        parser.error('Probabilities must be in [0, 1]')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mmdet3d.apis.train_stream import prepare_streaming_config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.samplers import GroupInBatchSampler
    cfg = prepare_streaming_config(load_config(args.config))
    cfg.seed = args.seed
    configured_cbgs = bool(cfg.data.train.get('sequence_cbgs', False))
    cfg.data.train.sequence_cbgs = True  # build metadata for before/after CBGS summary
    dataset = build_dataset(cfg.data.train)
    root = args.out_dir.expanduser() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    summary_dir = root / 'sampling_summary'; summary_dir.mkdir()
    cbgs_sampler = GroupInBatchSampler(dataset, 1, seed=args.seed)
    presence = dataset.sequence_class_presence
    repeats = np.bincount(cbgs_sampler.cbgs_group_indices, minlength=len(dataset.sequence_indices))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(range(len(repeats)), repeats); axes[0].set(title='Sequence CBGS repeats', xlabel='group ID', ylabel='draws')
    x = np.arange(len(dataset.CLASSES)); before = presence.mean(0); after = presence.T @ repeats / repeats.sum()
    axes[1].bar(x-.2, before, .4, label='original'); axes[1].bar(x+.2, after, .4, label='CBGS')
    axes[1].set_xticks(x); axes[1].set_xticklabels(dataset.CLASSES, rotation=70); axes[1].set_ylabel('Fraction of groups containing class'); axes[1].legend()
    fig.tight_layout(); fig.savefig(summary_dir / 'cbgs_comparison.png'); plt.close(fig)
    save_json(summary_dir / 'cbgs.json', dict(classes=list(dataset.CLASSES), repeats=repeats, before=before, after=after))
    # Actual configured schedule, distinct from controlled same-group comparisons.
    dataset.sequence_cbgs = configured_cbgs
    events = []
    overview = GroupInBatchSampler(dataset, args.slots, seed=args.seed, diagnostic_callback=events.append)
    batches = list(itertools.islice(iter(overview), args.steps))
    ids = np.asarray([[int(dataset.flag[r['idx']]) for r in b] for b in batches]).T
    fig, ax = plt.subplots(figsize=(16, 4)); ax.imshow(ids, aspect='auto', interpolation='nearest', cmap='tab20')
    for step, batch in enumerate(batches):
        for slot, request in enumerate(batch):
            if request['sequence_start']:ax.plot(step, slot, 'k|')
    ax.set(xlabel='Training step', ylabel='Global slot', title='Group IDs (colors may repeat); black marks = history reset')
    fig.tight_layout(); fig.savefig(summary_dir / 'slot_timeline.png'); plt.close(fig)
    save_json(summary_dir / 'schedule.json', dict(sequence_cbgs=configured_cbgs, batches=batches, skipped_events=events))
    active = [g for g, indices in enumerate(dataset.sequence_indices) if dataset.has_gt_flags[indices].any()]
    chosen = args.group_ids if args.group_ids is not None else np.random.RandomState(args.seed).choice(active, min(args.num_groups, len(active)), replace=False).tolist()
    if any(g not in active for g in chosen):raise ValueError('Selected group is invalid or has no filtered GT')
    normalizers = [t for t in dataset.pipeline.transforms if type(t).__name__ == 'ImageNormalize']
    if len(normalizers) != 1:raise ValueError('Expected exactly one ImageNormalize transform')
    mean, std = np.asarray(normalizers[0].mean), np.asarray(normalizers[0].std)
    variants = [('baseline', True, 0., 0.), ('frame_aug', False, 0., 0.),
                ('random_skip', True, args.skip_prob, 0.), ('reverse', True, 0., args.flip_prob),
                ('combined', False, args.skip_prob, args.flip_prob)]
    class FixedGroupSampler(GroupInBatchSampler):
        def _groups(self):
            yield from itertools.repeat(self.selected_group)
    dataset.sequence_cbgs = False
    save_json(root / 'manifest.json', dict(config=str(args.config.resolve()), seed=args.seed, selected_groups=chosen,
        variants=variants, trace_only=args.trace_only, max_frames=args.max_frames,
        notes=['Same groups across variants; forced group selection only for controlled panels.',
               'baseline/random_skip/reverse share first geometric augmentation draw.',
               'Worker randomness seeded by frame index for controlled comparison.',
               'No model/cache execution: history source and reset annotations are inferred from sampler requests.',
               'Normalization is undone for display; point display is subsampled only after pipeline processing.']))
    for group in chosen:
        indices = dataset.sequence_indices[group]
        directory = root / ('scene_%s_group_%04d' % (dataset.data_infos[indices[0]]['scene_token'], group)); directory.mkdir()
        scene = dataset.data_infos[indices[0]]['scene_token']
        scene_groups = [dict(group_id=g, indices=idxs) for g, idxs in enumerate(dataset.sequence_indices)
                        if dataset.data_infos[idxs[0]]['scene_token'] == scene]
        save_json(directory / 'manifest.json', dict(group_id=group, original_indices=indices, scene_groups=scene_groups))
        fig, ax = plt.subplots(figsize=(14, 2))
        offset = 0
        for entry in scene_groups:
            length = len(entry['indices'])
            ax.broken_barh([(offset, length)], (0, 1), facecolors='orange' if entry['group_id'] == group else 'lightgray', edgecolors='black')
            ax.text(offset + length / 2, .5, str(entry['group_id']), ha='center')
            offset += length
        ax.set(xlim=(0, offset), yticks=[], xlabel='Original frame position within scene', title='Scene split; labels = group IDs, selected group = orange')
        fig.tight_layout(); fig.savefig(directory / 'scene_split.png'); plt.close(fig)
        for name, consistent, skip, flip in variants:
            target = directory / name; target.mkdir()
            dataset.keep_consistent_seq_aug = consistent
            dataset.skip_prob = skip; dataset.sequence_flip_prob = flip
            trace = []
            sampler = FixedGroupSampler(dataset, 1, seed=args.seed + group, diagnostic_callback=trace.append)
            sampler.selected_group = group
            requests = []
            for batch in sampler:
                request = batch[0]
                if requests and request['sequence_instance'] != requests[0]['sequence_instance']:break
                requests.append(request)
            trace = [e for e in trace if e['sequence_instance'] == requests[0]['sequence_instance']]
            fig, ax = plt.subplots(figsize=(14, 2)); timeline(ax, indices, requests, trace)
            fig.tight_layout(); fig.savefig(target / 'timeline.png'); plt.close(fig)
            save_json(target / 'trace.json', dict(requests=requests, skipped_events=trace))
            previous = None
            for step, request in enumerate(requests):
                idx = request['idx']; info = dataset.data_infos[idx]
                dt = None if previous is None else (info['timestamp'] - previous['timestamp']) / 1e6
                position = indices.index(idx)
                title = '%s | group=%d step=%d frame=%d | dt=%s | start=%s direction=%d\n%s' % (
                    name, group, step, position, dt, request['sequence_start'], request['sequence_direction'], info['token'])
                if not args.trace_only and (args.max_frames == 0 or step < args.max_frames):
                    meta = draw_frame(dataset, request, target / ('step_%03d_frame_%03d.png' % (step, position)),
                                      title, indices, requests, trace, mean, std, args.lidar_limit, args.max_points)
                    save_json(target / ('step_%03d.json' % step), dict(current=meta, history_token=None if previous is None else previous['token'], dt=dt))
                previous = info
            print('Saved group %d / %s (%d emitted frames)' % (group, name, len(requests)), flush=True)
    print('Saved:', root)


if __name__ == '__main__':
    main()
