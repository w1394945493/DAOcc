"""Offline per-subsequence detection-class statistics for streaming DAOcc.

Matches NuScenesSequenceDataset ordering and split semantics. Splits ORIGINAL
frames first, then optionally excludes empty GT inside each group, like sampler.
Requires only NumPy; no images, point clouds, OCC arrays or model are loaded.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from analyze_cbgs_sampling import (add_common_arguments, class_stats, load_counts,
                                   write_csv, write_json)


def split_groups(infos, split):
    scenes = {}
    for i, info in enumerate(infos):
        if not info.get('scene_token'):
            raise ValueError('Missing scene_token at index %d' % i)
        scenes.setdefault(info['scene_token'], []).append(i)
    for scene, indices in scenes.items():
        times = [infos[i]['timestamp'] for i in indices]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Non-increasing timestamps: ' + scene)
    if split == 'all':
        return [[i] for i in range(len(infos))]
    groups = []
    for scene in sorted(scenes):
        indices = scenes[scene]
        step = math.ceil(len(indices) / split)
        groups.extend(indices[start:start + step] for start in range(0, len(indices), step))
    if len(groups) != len(scenes) * split:
        raise ValueError('Split produced %d groups, expected %d; same constraint as dataset.' %
                         (len(groups), len(scenes) * split))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--cbgs', action='store_true', help='Also sample whole subsequences using class-presence CBGS.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for CBGS draws and group-order shuffle.')
    parser.add_argument('--sequences-split-num', default='2', help='Scene split count K, or all for single frames.')
    parser.add_argument('--keep-empty-gt', action='store_true',
                        help='Include empty frames; default matches skip_empty_gt=True.')
    args = parser.parse_args()
    split = args.sequences_split_num
    if split != 'all':
        try:
            split = int(split)
        except ValueError:
            parser.error('--sequences-split-num must be a positive integer or all')
        if split < 1:
            parser.error('--sequences-split-num must be positive')
    infos, _, counts = load_counts(args)
    groups = split_groups(infos, split)
    eligible = counts.sum(axis=1) > 0
    records, detail, membership = [], [], []
    group_presence = np.zeros((len(groups), len(args.classes)), dtype=bool)
    retained = []
    for group_id, indices in enumerate(groups):
        used = [i for i in indices if args.keep_empty_gt or eligible[i]]
        retained.extend(used)
        scene = infos[indices[0]]['scene_token']
        record = dict(group_id=group_id, scene_token=scene, total_frames=len(indices),
                      retained_frames=len(used), skipped_frames=len(indices) - len(used),
                      first_token=infos[indices[0]]['token'], last_token=infos[indices[-1]]['token'],
                      first_timestamp=int(infos[indices[0]]['timestamp']),
                      last_timestamp=int(infos[indices[-1]]['timestamp']),
                      first_retained_token=infos[used[0]]['token'] if used else '',
                      last_retained_token=infos[used[-1]]['token'] if used else '')
        records.append(record)
        for position, i in enumerate(indices):
            membership.append(dict(group_id=group_id, position=position, dataset_index=i,
                                   token=infos[i]['token'], scene_token=scene,
                                   retained=bool(args.keep_empty_gt or eligible[i])))
        for c, stats in enumerate(class_stats(counts[used], args.classes)):
            present = stats['frames_with_class']
            group_presence[group_id, c] = present > 0
            detail.append(dict(group_id=group_id, scene_token=scene,
                               class_name=stats['class_name'], present=present > 0,
                               retained_frames=len(used), frames_with_class=present,
                               frame_fraction=stats['frame_fraction'], boxes=stats['boxes'],
                               boxes_per_retained_frame=stats['boxes'] / len(used) if used else 0.,
                               boxes_per_positive_frame=stats['boxes'] / present if present else 0.,
                               max_boxes_in_frame=int(counts[used, c].max()) if used else 0))
    active = sum(r['retained_frames'] > 0 for r in records)
    class_rows = class_stats(counts[retained], args.classes)
    for c, row in enumerate(class_rows):
        number = int(group_presence[:, c].sum())
        row.update(groups_with_class=number, total_groups=len(groups), active_groups=active,
                   active_group_fraction=number / active if active else 0.)
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'sequences.csv', records)
    write_csv(out / 'sequence_classes.csv', detail)
    write_csv(out / 'frame_membership.csv', membership)
    write_csv(out / 'class_statistics.csv', class_rows)
    cooccurrence = group_presence.astype(np.int64).T @ group_presence.astype(np.int64)
    write_csv(out / 'class_cooccurrence.csv', [dict(class_name=name, **{
        other: int(cooccurrence[c, j]) for j, other in enumerate(args.classes)})
        for c, name in enumerate(args.classes)])
    lengths = [r['retained_frames'] for r in records]
    write_json(out / 'summary.json', dict(
        ann_file=str(Path(args.ann_file).expanduser().resolve()), classes=args.classes,
        use_valid_flag=not args.no_valid_flag, bev_range=args.bev_range,
        sequences_split_num=split, skip_empty_gt=not args.keep_empty_gt,
        original_frames=len(infos), retained_frames=len(retained),
        skipped_frames=len(infos) - len(retained),
        scenes=len({i['scene_token'] for i in infos}), groups=len(groups),
        active_groups=active, empty_groups=len(groups) - active,
        retained_length_percentiles=dict(zip(['min', 'p25', 'p50', 'p75', 'max'],
                                             np.percentile(lengths, [0, 25, 50, 75, 100]).tolist())),
        notes=['Detection classes only, not occupancy voxel classes.',
               'Counts use validity, known classes and strict XY center range before stochastic geometry.',
               'Split original scene frames first, then skip empty GT without regrouping.',
               'Each group counted once: this is not a simulated finite multi-GPU training run.',
               'Box totals count repeated observations across frames, not unique tracked instances.',
               'Frame fraction denominator is retained frames; group fraction uses nonempty retained groups.']))
    print('Scenes: %d; groups: %d; active: %d; frames: %d -> %d' %
          (len({i['scene_token'] for i in infos}), len(groups), active, len(infos), len(retained)))
    print('%-23s %10s %12s %12s' % ('class', 'groups', 'frames', 'boxes'))
    for row in class_rows:
        print('%-23s %10d %12d %12d' % (row['class_name'], row['groups_with_class'],
                                       row['frames_with_class'], row['boxes']))
    print('Saved:', out)
    if args.cbgs:
        analyze_cbgs(args, groups, eligible, counts, group_presence, records, class_rows, out)


#* (11.13) 离线子序列级 CBGS；不修改训练 sampler，段内保留原时间顺序。
def analyze_cbgs(args, groups, eligible, counts, presence, records, before, out):
    pools = [np.flatnonzero(presence[:, c]) for c in range(len(args.classes))]
    missing = [name for name, pool in zip(args.classes, pools) if not len(pool)]
    if missing:
        raise ValueError('CBGS cannot balance absent classes: %s' % missing)
    rng = np.random.RandomState(args.seed)
    relations = sum(len(pool) for pool in pools)
    draws, pool_rows = [], []
    for name, pool in zip(args.classes, pools):
        ratio = (1. / len(args.classes)) / (len(pool) / relations)
        size = int(len(pool) * ratio)
        draws.append(rng.choice(pool, size=size, replace=True))
        pool_rows.append(dict(class_name=name, pool_groups=len(pool),
                              sampling_ratio=ratio, drawn_groups=size))
    sampled = np.concatenate(draws)
    rng.shuffle(sampled)
    repeats = np.bincount(sampled, minlength=len(groups))
    frame_weights = np.zeros(len(counts), dtype=np.int64)
    used_groups = []
    for group_id, indices in enumerate(groups):
        used = [i for i in indices if args.keep_empty_gt or eligible[i]]
        used_groups.append(used)
        frame_weights[used] = repeats[group_id]
    after = class_stats(counts, args.classes, frame_weights)
    sampled_presence = presence.astype(np.int64).T @ repeats
    comparisons = []
    for c, name in enumerate(args.classes):
        after[c].update(groups_with_class=int(sampled_presence[c]),
                        total_groups=len(sampled), active_groups=len(sampled),
                        active_group_fraction=float(sampled_presence[c] / len(sampled)))
        row = dict(class_name=name)
        for prefix, stats in [('before', before[c]), ('after', after[c])]:
            row.update({prefix + '_' + k: v for k, v in stats.items() if k != 'class_name'})
        comparisons.append(row)
    directory = out / 'cbgs'
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / 'class_statistics.csv', comparisons)
    write_csv(directory / 'sampling_pools.csv', pool_rows)
    write_csv(directory / 'sequence_repeats.csv', [dict(
        group_id=i, scene_token=r['scene_token'], retained_frames=r['retained_frames'],
        repeats=int(repeats[i]), sampled_frames=int(repeats[i]) * r['retained_frames'])
        for i, r in enumerate(records)])
    # 每次抽取均为独立实例；通过有序帧索引和 offset 可重建每段，不跨重复段接续缓存。
    offsets = [0]
    flattened = []
    instances = []
    for instance, group_id in enumerate(sampled):
        used = used_groups[group_id]
        flattened.extend(used)
        offsets.append(len(flattened))
        instances.append(dict(sequence_instance=instance, group_id=int(group_id),
                              scene_token=records[group_id]['scene_token'], frames=len(used),
                              frame_offset_start=offsets[-2], frame_offset_end=offsets[-1],
                              reset_history=True))
    np.save(directory / 'sampled_group_indices.npy', sampled)
    np.save(directory / 'sampled_frame_indices.npy', np.asarray(flattened, dtype=np.int64))
    np.save(directory / 'sequence_offsets.npy', np.asarray(offsets, dtype=np.int64))
    write_csv(directory / 'sampled_sequences.csv', instances)
    original_frames = sum(len(g) for g in used_groups)
    summary = dict(seed=args.seed, classes=args.classes, original_groups=len(groups),
                   original_retained_frames=original_frames,
                   sampled_groups=len(sampled), sampled_frames=int(frame_weights.sum()),
                   unique_sampled_groups=int((repeats > 0).sum()),
                   unsampled_groups=int((repeats == 0).sum()),
                   unique_sampled_frames=int((frame_weights > 0).sum()),
                   max_group_repeats=int(repeats.max()),
                   frame_expansion_factor=float(frame_weights.sum() / original_frames),
                   ann_file=str(Path(args.ann_file).expanduser().resolve()),
                   sequences_split_num=args.sequences_split_num,
                   skip_empty_gt=not args.keep_empty_gt,
                   use_valid_flag=not args.no_valid_flag, bev_range=args.bev_range,
                   notes=['Class pools use filtered GT presence, not box count or positive-frame density.',
                          'Whole groups are sampled with replacement, then shuffled; within-group order is preserved.',
                          'Groups with no target class are absent from every pool and are never sampled.',
                          'Each occurrence is a new sequence instance requiring history reset.',
                          'Statistics describe one sampled list, not multi-GPU scheduling or actual training updates.',
                          'Original statistics remain in the parent directory; all ratios are fractions, not percentages.'])
    write_json(directory / 'summary.json', summary)
    print('\nSequence CBGS: groups %d -> %d; frames %d -> %d; unique groups %d' %
          (len(groups), len(sampled), original_frames, frame_weights.sum(), (repeats > 0).sum()))
    print('%-23s %9s %9s %10s %10s %11s %11s' %
          ('class', 'groups_in', 'groups_out', 'frames_in', 'frames_out', 'boxes_in', 'boxes_out'))
    for a, b in zip(before, after):
        print('%-23s %9d %9d %10d %10d %11d %11d' %
              (a['class_name'], a['groups_with_class'], b['groups_with_class'],
               a['frames_with_class'], b['frames_with_class'], a['boxes'], b['boxes']))
    print('CBGS saved:', directory)


if __name__ == '__main__':
    main()
