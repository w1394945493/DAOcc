"""Offline reproduction of DAOcc CBGSDataset sampling (NumPy only).

Counts detection annotations, not OCC voxels or distinct tracked instances.
Defaults match use_valid_flag=True and the current identity geometric augmentation
with symmetric XY flips. Does not simulate worker-side random empty-GT replacement.
The seed reproduces an isolated CBGS draw, not a historical run's RNG state.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
DEFAULT_ANN = '/home/wangyushen/Data/Datasets/nuscenes_anns/anns/daocc/nuscenes_infos_train.pkl'


def add_common_arguments(parser):
    parser.add_argument('--ann-file', default=DEFAULT_ANN)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--classes', nargs='+', default=CLASSES)
    parser.add_argument('--bev-range', nargs=4, type=float,
                        default=[-54., -54., 54., 54.],
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    parser.add_argument('--no-valid-flag', action='store_true',
                        help='Match use_valid_flag=False: CBGS uses all boxes; training uses num_lidar_pts>0.')


def load_counts(args):
    """Return timestamp-sorted infos, CBGS-box counts and filtered training counts."""
    if not args.classes or len(set(args.classes)) != len(args.classes):
        raise ValueError('--classes must be nonempty and unique')
    xmin, ymin, xmax, ymax = args.bev_range
    if not np.isfinite(args.bev_range).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError('Invalid --bev-range')
    with Path(args.ann_file).expanduser().open('rb') as handle:
        payload = pickle.load(handle)
    infos = sorted(payload['infos'], key=lambda info: info['timestamp'])
    if not infos:
        raise ValueError('Empty annotation dataset')
    cbgs = np.zeros((len(infos), len(args.classes)), dtype=np.int64)
    filtered = np.zeros_like(cbgs)
    for index, info in enumerate(infos):
        names = np.asarray(info['gt_names'], dtype=str)
        # LiDAR boxes store float32 centers in the actual dataset pipeline.
        boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
        if len(boxes) != len(names):
            raise ValueError('GT boxes/names length mismatch at index %d' % index)
        if args.no_valid_flag:
            sampling_mask = np.ones(len(names), dtype=bool)
            train_mask = np.asarray(info['num_lidar_pts']) > 0
        else:
            sampling_mask = np.asarray(info['valid_flag'], dtype=bool)
            train_mask = sampling_mask.copy()
        if sampling_mask.shape != names.shape or train_mask.shape != names.shape:
            raise ValueError('Invalid GT mask shape at index %d' % index)
        if len(names):
            # Matches LiDARInstance3DBoxes.in_range_bev: strict XY center bounds.
            train_mask = train_mask & (boxes[:, 0] > xmin) & (boxes[:, 0] < xmax)
            train_mask &= (boxes[:, 1] > ymin) & (boxes[:, 1] < ymax)
        for c, name in enumerate(args.classes):
            cbgs[index, c] = np.count_nonzero(sampling_mask & (names == name))
            filtered[index, c] = np.count_nonzero(train_mask & (names == name))
    return infos, cbgs, filtered


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def class_stats(counts, classes, weights=None):
    weights = np.ones(len(counts), dtype=np.int64) if weights is None else weights
    frames = int(weights.sum())
    present = (counts > 0).astype(np.int64).T @ weights
    boxes = counts.T @ weights
    relations = int(present.sum())
    total_boxes = int(boxes.sum())
    return [dict(class_name=name, frames_with_class=int(present[c]),
                 frame_fraction=float(present[c] / frames) if frames else 0.,
                 frame_class_relation_fraction=float(present[c] / relations) if relations else 0.,
                 boxes=int(boxes[c]),
                 box_fraction=float(boxes[c] / total_boxes) if total_boxes else 0.)
            for c, name in enumerate(classes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    infos, counts, filtered = load_counts(args)
    pools = [np.flatnonzero(counts[:, c]) for c in range(len(args.classes))]
    if any(len(pool) == 0 for pool in pools):
        missing = [name for name, pool in zip(args.classes, pools) if not len(pool)]
        raise ValueError('Original CBGS cannot sample absent classes: %s' % missing)
    duplicated = sum(len(pool) for pool in pools)
    rng = np.random.RandomState(args.seed)
    draws, sampling_rows = [], []
    for name, pool in zip(args.classes, pools):
        # Preserve the original floating-point expression and integer truncation.
        ratio = (1. / len(args.classes)) / (len(pool) / duplicated)
        size = int(len(pool) * ratio)
        draws.append(rng.choice(pool, size=size, replace=True))
        sampling_rows.append(dict(class_name=name, pool_frames=len(pool),
                                  sampling_ratio=ratio, drawn_indices=size))
    indices = np.concatenate(draws)
    repeats = np.bincount(indices, minlength=len(infos))
    eligible = filtered.sum(axis=1) > 0
    before = class_stats(counts, args.classes)
    after = class_stats(counts, args.classes, repeats)
    filtered_before = class_stats(filtered, args.classes)
    filtered_after = class_stats(filtered, args.classes, repeats)
    rows = []
    for c, name in enumerate(args.classes):
        row = dict(class_name=name)
        for prefix, stats in [('original', before), ('cbgs', after),
                              ('original_filtered', filtered_before),
                              ('cbgs_filtered', filtered_after)]:
            row.update({prefix + '_' + k: v for k, v in stats[c].items() if k != 'class_name'})
        rows.append(row)
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(ann_file=str(Path(args.ann_file).expanduser().resolve()), seed=args.seed,
                   classes=args.classes, use_valid_flag=not args.no_valid_flag,
                   bev_range=args.bev_range, original_frames=len(infos),
                   sampled_frames=len(indices), expansion_factor=len(indices) / len(infos),
                   unique_sampled_frames=int((repeats > 0).sum()),
                   unsampled_frames=int((repeats == 0).sum()), max_frame_repeats=int(repeats.max()),
                   original_empty_after_filters=int((~eligible).sum()),
                   sampled_empty_after_filters=int(repeats[~eligible].sum()),
                   original_scenes=len({i['scene_token'] for i in infos}),
                   sampled_scenes=len({infos[i]['scene_token'] for i in np.flatnonzero(repeats)}),
                   sampling_pools=sampling_rows,
                   notes=['CBGS sampling unit is one frame, not one full scene.',
                          'Filtered counts use strict XY center bounds, before stochastic geometry.',
                          'Counts do not simulate random replacement of empty samples by workers.',
                          'Box counts include repeated objects across frames and repeated sampled frames.',
                          'A fixed seed reproduces this audit, not the RNG state of an earlier training run.',
                          'Frame fractions can sum above one because classes co-occur.'])
    write_json(out / 'summary.json', summary)
    write_csv(out / 'class_statistics.csv', rows)
    write_csv(out / 'sampling_pools.csv', sampling_rows)
    write_csv(out / 'frame_repeats.csv', [dict(dataset_index=i, token=info['token'],
              scene_token=info['scene_token'], repeats=int(repeats[i]),
              has_filtered_gt=bool(eligible[i])) for i, info in enumerate(infos)])
    np.save(out / 'sample_indices.npy', indices)
    print('Frames: %d -> %d; unique sampled: %d; rejected candidates: %d' %
          (len(infos), len(indices), (repeats > 0).sum(), repeats[~eligible].sum()))
    print('%-23s %12s %12s %12s %12s' % ('class', 'frames_before', 'frames_after', 'boxes_before', 'boxes_after'))
    for a, b in zip(before, after):
        print('%-23s %12d %12d %12d %12d' % (a['class_name'], a['frames_with_class'],
              b['frames_with_class'], a['boxes'], b['boxes']))
    print('Saved:', out)


if __name__ == '__main__':
    main()
