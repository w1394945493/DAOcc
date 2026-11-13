"""Build configured DAOcc datasets and compare frame/sequence CBGS on CPU.

No __getitem__, sensor loading, model construction or training is performed.
Epochs for the infinite sequence sampler mean one resampled-list frame budget,
not runner epochs. Empty-frame random replacement in ordinary training is not
simulated; it does not change the nominal number of optimizer steps.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_cbgs_sampling import class_stats, write_csv, write_json


def load_config(path):
    from torchpack.utils.config import Config as TorchpackConfig
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    # Independent instances: streaming settings must not leak into baseline.
    values = TorchpackConfig()
    values.load(str(path), recursive=True)
    return Config(recursive_eval(values), filename=str(path))


def filtered_counts(dataset):
    from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter, ObjectNameFilter
    filters = [t for t in dataset.pipeline.transforms
               if isinstance(t, (ObjectRangeFilter, ObjectNameFilter))]
    counts = np.zeros((len(dataset.data_infos), len(dataset.CLASSES)), dtype=np.int64)
    for i in range(len(counts)):
        data = deepcopy(dataset.get_ann_info(i))
        for transform in filters:
            data = transform(data)
        labels = np.asarray(data['gt_labels_3d'], dtype=np.int64)
        counts[i] = np.bincount(labels[labels >= 0], minlength=len(dataset.CLASSES))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = ROOT / 'configs/nuscenes/occ3d'
    parser.add_argument('--base-config', type=Path, default=config_dir / 'daocc_occ3d_nus_w_mask_inin_spconv2_voxel.yaml')
    parser.add_argument('--stream-config', type=Path, default=config_dir / 'daocc_occ3d_nus_w_mask_inin_spconv2_voxel_stream_cache.yaml')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--raw-epochs', type=int, default=24)
    parser.add_argument('--frame-cbgs-epochs', type=int, default=6)
    parser.add_argument('--sequence-cbgs-epochs', type=int, default=5)
    args = parser.parse_args()
    if min(args.raw_epochs, args.frame_cbgs_epochs, args.sequence_cbgs_epochs) < 1:
        parser.error('Epoch budgets must be positive')

    import torch
    from mmdet3d.datasets import build_dataset
    from mmdet3d.apis.train_stream import prepare_streaming_config
    from mmdet3d.datasets.samplers import GroupInBatchSampler
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_cfg = load_config(args.base_config)
    if base_cfg.data.train.type != 'CBGSDataset':
        raise ValueError('Expected baseline train config to use CBGSDataset')
    frame_dataset = build_dataset(deepcopy(base_cfg.data.train))
    if not frame_dataset.resample:
        raise ValueError('Baseline config disables CBGS resampling')
    raw_dataset = frame_dataset.dataset
    print('Built baseline dataset and its actual CBGSDataset wrapper.', flush=True)

    stream_cfg = load_config(args.stream_config)
    stream_cfg.seed = args.seed
    stream_cfg = prepare_streaming_config(stream_cfg)
    # Explicit scenario requested by this audit; original YAML is never changed.
    stream_cfg.data.train.sequence_cbgs = True
    stream_dataset = build_dataset(deepcopy(stream_cfg.data.train))
    sampler = GroupInBatchSampler(stream_dataset, batch_size=1, rank=0, world_size=1, seed=args.seed)
    if tuple(raw_dataset.CLASSES) != tuple(stream_dataset.CLASSES):
        raise ValueError('Dataset class orders differ')
    if [i['token'] for i in raw_dataset.data_infos] != [i['token'] for i in stream_dataset.data_infos]:
        raise ValueError('Dataset frame lists differ; cannot compare the same training set')
    raw_counts = filtered_counts(raw_dataset)
    stream_counts = filtered_counts(stream_dataset)
    if not np.array_equal(raw_counts, stream_counts):
        raise ValueError('Configured annotation filters differ; refusing an unequal GT comparison')
    frame_indices = np.asarray(frame_dataset.sample_indices, dtype=np.int64)
    frame_weights = np.bincount(frame_indices, minlength=len(raw_counts))
    group_indices = sampler.cbgs_group_indices
    group_weights = np.bincount(group_indices, minlength=len(stream_dataset.sequence_indices))
    sequence_weights = np.zeros(len(stream_counts), dtype=np.int64)
    for g, indices in enumerate(stream_dataset.sequence_indices):
        used = [i for i in indices if not stream_dataset.skip_empty_gt or stream_dataset.has_gt_flags[i]]
        sequence_weights[used] = group_weights[g]
    if int(sequence_weights.sum()) != sampler.size:
        raise RuntimeError('Sampler length disagrees with emitted frame statistics')

    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    classes = list(raw_dataset.CLASSES)
    scenarios = [
        ('original', np.ones(len(raw_counts), dtype=np.int64), args.raw_epochs),
        ('frame_cbgs', frame_weights, args.frame_cbgs_epochs),
        ('sequence_cbgs', sequence_weights, args.sequence_cbgs_epochs)]
    budgets, statistics = [], []
    empty = raw_counts.sum(axis=1) == 0
    for name, weights, epochs in scenarios:
        frames = int(weights.sum())
        budgets.append(dict(scenario=name, frames_per_pass=frames,
                            unique_frames=int((weights > 0).sum()),
                            empty_gt_entries=int(weights[empty].sum()),
                            epochs=epochs, gpus=1, batch_size=1,
                            steps_per_epoch=frames, total_steps=frames * epochs))
        for row in class_stats(raw_counts, classes, weights):
            statistics.append(dict(scenario=name, **row))
    write_csv(out / 'training_steps.csv', budgets)
    write_csv(out / 'class_statistics.csv', statistics)
    write_csv(out / 'frame_repeats.csv', [dict(
        dataset_index=i, token=info['token'], scene_token=info['scene_token'],
        has_filtered_gt=bool(not empty[i]), frame_cbgs_repeats=int(frame_weights[i]),
        sequence_cbgs_repeats=int(sequence_weights[i]))
        for i, info in enumerate(raw_dataset.data_infos)])
    write_csv(out / 'sequence_repeats.csv', [dict(
        group_id=g, original_frames=len(indices), repeats=int(group_weights[g]),
        retained_frames=sum(not stream_dataset.skip_empty_gt or bool(stream_dataset.has_gt_flags[i]) for i in indices))
        for g, indices in enumerate(stream_dataset.sequence_indices)])
    np.save(out / 'frame_cbgs_indices.npy', frame_indices)
    np.save(out / 'sequence_cbgs_group_indices.npy', group_indices)
    write_json(out / 'summary.json', dict(
        base_config=str(args.base_config.resolve()), stream_config=str(args.stream_config.resolve()),
        seed=args.seed, classes=classes, sequence_cbgs=True,
        sequences_split_num=stream_dataset.sequences_split_num,
        skip_empty_gt=stream_dataset.skip_empty_gt,
        original_groups=len(stream_dataset.sequence_indices),
        sampled_groups=len(group_indices), budgets=budgets,
        notes=['Builds real configured datasets, CBGSDataset and GroupInBatchSampler; no sensor pipeline is executed.',
               'Both class tables use actual get_ann_info plus ObjectRangeFilter/ObjectNameFilter, before stochastic geometry.',
               'Baseline CBGS pools retain the original get_cat_ids semantics, before range filtering.',
               'Original/frame-CBGS empty candidates are counted; worker random replacement is not simulated.',
               'Counts are detection boxes, not OCC voxels or distinct tracked instances.',
               'Sequence epoch means one sampled-list frame budget at BS=1; its runner remains iteration-based.',
               'Total steps are nominal iterations, not guaranteed successful AMP optimizer updates.',
               'Seed controls this audit, not a past training RNG state.']))
    print('\nSingle GPU, batch size=1')
    print('%-18s %12s %8s %14s' % ('scenario', 'steps/epoch', 'epochs', 'total_steps'))
    for row in budgets:
        print('%-18s %12d %8d %14d' % (row['scenario'], row['steps_per_epoch'], row['epochs'], row['total_steps']))
    print('\n%-23s %14s %14s %14s' % ('class (frame %)', 'original', 'frame_cbgs', 'sequence_cbgs'))
    for c, name in enumerate(classes):
        values = [statistics[k * len(classes) + c]['frame_fraction'] * 100 for k in range(3)]
        print('%-23s %13.2f%% %13.2f%% %13.2f%%' % (name, *values))
    print('Saved:', out)


if __name__ == '__main__':
    main()
