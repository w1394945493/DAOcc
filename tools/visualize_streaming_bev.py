#!/usr/bin/env python3
"""检查单卡/多卡训练及训练中流式评估的帧顺序、历史 BEV 缓存，并保存逐帧对照图。

train 使用训练 Dataset/增强和 model.train()；val 使用训练中流式验证 Dataset/增强和 eval()。
通过 --split train/val/both 选择处理范围，默认 both，两边各执行 steps。
train 使用指定每卡 BS，val 按 StreamOcc 每卡固定 BS=1；no_grad，不更新参数。
多卡使用 torchrun --standalone --nproc_per_node=2 启动，输出增加 rank_00/rank_01 层。
训练态 BN 统计会在本进程内变化；不保存权重。
每幅图依次显示原始历史、对齐到当前坐标系的历史、当前 BEV；灰色为无效历史或越界。
训练几何增强由 data.train.dataset.keep_consistent_seq_aug 控制，默认子序列内一致。
示例（在仓库根目录、DAOcc 环境中执行）：
  python tools/visualize_streaming_bev.py CONFIG CHECKPOINT --out-dir OUTPUT \
      --batch-size 4 --steps 24 --device 0
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import shutil
import traceback
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config, DictAction
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from torchpack.utils.config import Config as TorchpackConfig

from mmdet3d.apis.train_stream import (prepare_streaming_config, build_stream_dataloader,
                                         build_stream_val_dataloader)
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval, convert_sync_batchnorm


def signature(feature):
    # 完整张量的 FP32 数值摘要用于对应检查，不以热力图相似性代替缓存验证。
    array = feature.detach().float().cpu().contiguous().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError('BEV contains NaN/Inf')
    return hashlib.sha256(array.tobytes()).hexdigest(), np.abs(array).mean(axis=0).T


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(value))


def draw_triplet(path, current, history, aligned, mask, record, point_cloud_range):
    arrays = [current.ravel()]
    if history is not None:
        arrays.append(history.ravel())
        if mask.any():
            arrays.append(aligned[mask])
    values = np.concatenate(arrays)
    low, high = np.percentile(values, [1, 99]).tolist()
    high = max(high, low + 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    try:
        xmin, ymin, _, xmax, ymax, _ = point_cloud_range
        extent = (xmin, xmax, ymin, ymax)
        cmap = deepcopy(plt.get_cmap('viridis'))
        cmap.set_bad('#dddddd')
        style = dict(origin='lower', cmap=cmap, vmin=low, vmax=high, extent=extent)
        old = record['history_source']
        if history is None:
            for ax in axes[:2]:
                ax.set_facecolor('#eeeeee')
                ax.text(.5, .5, 'NO VALID HISTORY\n' + record['history_reason'],
                        ha='center', va='center', transform=ax.transAxes)
            axes[0].set_title('Raw history BEV')
            axes[1].set_title('Aligned history BEV')
        else:
            axes[0].imshow(history, **style)
            axes[0].set_title(f"Raw history: {old['token'][:12]}\nt={old['timestamp']:.6f}")
            axes[1].imshow(np.ma.array(aligned, mask=~mask), **style)
            axes[1].set_title(f"History warped to current\nvalid area={record['warp']['valid_fraction']:.1%}")
        im = axes[2].imshow(current, **style)
        meta = record['current']
        axes[2].set_title(f"Current: {meta['token'][:12]}\nt={meta['timestamp']:.6f}")
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.set_xlabel('LiDAR X (m)')
            ax.set_ylabel('LiDAR Y (m)')
        fig.colorbar(im, ax=axes.tolist(), shrink=.75, label='mean(abs(BEV channels)); shared scale')
        fig.suptitle(f"Step {record['step']} | slot {meta['slot']} | scene {meta['scene_token']}\n"
                     f"group {record['group_id']} | frame {record['position'] + 1}/{record['group_length']} | "
                     f"start={meta['sequence_start']} | history_valid={record['history_valid']} | "
                     f"aug={meta['augmentation_mode']}")
        fig.text(.5, .02, 'Left: previous augmented LiDAR frame. Middle/right: current augmented LiDAR frame. '
                 'Planar ego-motion warp; moving objects may differ.', ha='center', fontsize=9)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches='tight')
    finally:
        plt.close(fig)
    return [low, high]


def check_alignment(warper, old, current, transform, grid, nx, ny):
    """Independently check transform and five grid positions in NumPy (FP64)."""
    old_global = np.asarray(old['ego2global']) @ np.asarray(old['lidar2ego'])
    cur_global = np.asarray(current['ego2global']) @ np.asarray(current['lidar2ego'])
    expected = (np.asarray(old['lidar_aug_matrix']) @ np.linalg.inv(old_global)
                @ cur_global @ np.linalg.inv(np.asarray(current['lidar_aug_matrix'])))
    actual = transform.detach().cpu().numpy()
    if not np.allclose(actual, expected, atol=1e-7, rtol=1e-7):
        raise AssertionError('BEV current-to-history transform differs from metadata composition')
    xmin, ymin, _, xmax, ymax, _ = warper.point_cloud_range
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    indices = np.array([[0, 0], [0, ny - 1], [nx - 1, 0], [nx - 1, ny - 1], [nx // 2, ny // 2]])
    points = np.stack((xmin + (indices[:, 0] + .5) * dx,
                       ymin + (indices[:, 1] + .5) * dy,
                       np.full(5, warper.plane_z), np.ones(5)), axis=-1)
    source = points @ expected.T
    ix = (source[:, 0] - (xmin + .5 * dx)) / dx
    iy = (source[:, 1] - (ymin + .5 * dy)) / dy
    expected_grid = np.stack((2 * iy / (ny - 1) - 1, 2 * ix / (nx - 1) - 1), axis=-1)
    actual_grid = grid.detach().cpu().numpy()[indices[:, 0], indices[:, 1]]
    error = float(np.max(np.abs(actual_grid - expected_grid)))
    if not np.allclose(actual_grid, expected_grid, atol=2e-5, rtol=1e-5):
        raise AssertionError(f'BEV grid axes/cell centers disagree with geometry: {error}')
    return dict(current_to_history=actual.tolist(), metadata_transform_matches=True,
                sampled_grid_matches=True, sampled_grid_max_abs_error=error,
                plane_z=warper.plane_z)


class CacheObserver:
    def __init__(self, model, dataset, root, slot_offset=0):
        self.model, self.root = model, root
        self.slot_offset = slot_offset
        self.augmentation_mode = ('validation' if dataset.test_mode else
                                  'sequence' if dataset.keep_consistent_seq_aug else 'frame')
        self.lookup = {idx: (group_id, position, len(indices))
                       for group_id, indices in enumerate(dataset.sequence_indices)
                       for position, idx in enumerate(indices)}
        self.previous = {}
        self.records = []
        self.step = 0
        self.pending = []
        self.transitions = {}

    def capture(self, current, aligned_history, valid):
        #* (11.12) 模型回调现在收到对齐历史；原始历史仍从缓存读取，用于逐帧哈希检查。
        history = self.model.bev_cache.bev
        alignment = self.model._history_alignment
        metas = self.model._stream_metas
        if len(metas) != current.shape[0]:
            raise AssertionError('Metadata and BEV batch sizes differ')
        if self.pending:
            raise AssertionError('Expected exactly one BEV capture per forward')
        if history is not None and (history.requires_grad or history.grad_fn is not None):
            raise AssertionError('History retains a computation graph')
        for i, meta in enumerate(metas):
            if meta['slot'] != self.slot_offset + i:
                raise AssertionError('Input order does not match this rank/global slot')
            group, position, length = self.lookup[meta['dataset_index']]
            #* (11.13) 首帧边界随读取方向变化，跳帧允许位置间隔大于一。
            direction = meta.get('sequence_direction', 1)
            if not self.model._skip_empty_gt_visual_check and bool(meta['sequence_start']) != (position == (0 if direction == 1 else length - 1)):
                raise AssertionError('Sequence-start flag does not match dataset group boundary')
            previous = self.previous.get(i)
            expected = previous is not None and not meta['sequence_start']
            #* (11.12) 检查同一子序列的显式参数与最终增强矩阵；逐帧模式只记录差异。
            # 独立采样也可能碰巧相同，因此 frame 模式不强制要求相邻帧增强不同。
            if meta['augmentation_mode'] != self.augmentation_mode:
                raise AssertionError('Dataset augmentation mode does not match observer')
            augmentation_check = None
            if expected:
                old_meta = previous['current']
                same_parameters = meta['aug_config'] == old_meta['aug_config']
                same_matrices = {}
                for key in ('lidar_aug_matrix', 'img_aug_matrix', 'occ_aug_matrix'):
                    if key in meta:
                        same_matrices[key] = bool(np.allclose(
                            np.asarray(meta[key]), np.asarray(old_meta[key]), atol=1e-6, rtol=1e-6))
                augmentation_check = dict(parameters_match_previous=same_parameters,
                                          matrices_match_previous=same_matrices)
                if self.augmentation_mode == 'sequence' and (
                        not same_parameters or not all(same_matrices.values())):
                    raise AssertionError(f'Slot {i}: augmentation changed inside the same subsequence')
            actual = history is not None and valid is not None and bool(valid[i].item())
            if actual != expected:
                raise AssertionError(f'Step {self.step}, slot {i}: unexpected history validity')
            history_heat = None
            history_hash = None
            aligned_heat, pixel_mask = None, None
            warp_record = None
            if expected:
                old = previous['current']
                if not (old['scene_token'] == meta['scene_token']
                        and old['sequence_instance'] == meta['sequence_instance']
                        and previous['group_id'] == group
                        and old.get('sequence_direction', 1) == direction
                        and ((position - previous['position']) * direction > 0
                             if self.model._skip_empty_gt_visual_check or self.model._temporal_skip_visual_check
                             else (position - previous['position']) * direction == 1)
                        and (meta['timestamp'] - old['timestamp']) * direction > 0):
                    raise AssertionError(f'Slot {i}: frames are not consecutive in the same subsequence')
                if self.model.bev_cache.metas[i] != old:
                    raise AssertionError(f'Slot {i}: cache source metadata mismatch')
                history_hash, history_heat = signature(history[i])
                if history_hash != previous['cached_sha256']:
                    raise AssertionError(f'Slot {i}: history does not match the previous cached output')
                if alignment is None or aligned_history is None:
                    raise AssertionError('Valid history was not aligned by the model')
                if aligned_history.requires_grad or aligned_history.grad_fn is not None:
                    raise AssertionError('Aligned history retains a computation graph')
                aligned_hash, aligned_heat = signature(aligned_history[i])
                pixel_mask = alignment['valid_mask'][i].cpu().numpy().T
                invalid = ~alignment['valid_mask'][i]
                #* (11.12) StreamOcc 零填充允许边界插值部分有效，仅旧掩码模式要求全零。
                if (self.model.bev_warp.mask_outside
                        and torch.count_nonzero(aligned_history[i][:, invalid]).item()):
                    raise AssertionError('Warped pixels outside historical support are not zero')
                warp_record = check_alignment(
                    self.model.bev_warp, old, meta, alignment['current_to_history'][i],
                    alignment['grid'][i], current.shape[2], current.shape[3])
                warp_record.update(aligned_sha256=aligned_hash,
                                   valid_fraction=float(pixel_mask.mean()))
            elif history is not None and torch.count_nonzero(history[i]).item() != 0:
                raise AssertionError(f'Slot {i}: invalid history was not zeroed')
            if not actual and aligned_history is not None:
                if (torch.count_nonzero(aligned_history[i]).item()
                        or alignment['valid_mask'][i].any().item()):
                    raise AssertionError(f'Slot {i}: invalid history produced valid warped pixels')
            if previous is not None and meta['sequence_start']:
                if (meta['sequence_instance'] <= previous['current']['sequence_instance']
                        or (not self.model._skip_empty_gt_visual_check
                            and meta['sequence_instance'] != previous['current']['sequence_instance'] + 1)):
                    raise AssertionError(f'Slot {i}: sequence instance did not advance')
                self.transitions[i] = self.transitions.get(i, 0) + 1
            current_hash, current_heat = signature(current[i])
            if pixel_mask is not None and pixel_mask.any():
                # Descriptive only: appearance/augmentation/dynamic motion can change features.
                # Lower heatmap error is NOT a correctness assertion for temporal alignment.
                warp_record['raw_heat_mae_on_valid_area'] = float(
                    np.abs(history_heat[pixel_mask] - current_heat[pixel_mask]).mean())
                warp_record['aligned_heat_mae_on_valid_area'] = float(
                    np.abs(aligned_heat[pixel_mask] - current_heat[pixel_mask]).mean())
            record = dict(step=self.step, local_slot=i, current=deepcopy(meta), group_id=group, position=position,
                          group_length=length, history_valid=actual,
                          history_reason=('valid' if actual else 'first batch' if previous is None else 'subsequence start'),
                          history_source=previous['current'] if actual else None,
                          history_sha256=history_hash, current_sha256=current_hash,
                          augmentation_check=augmentation_check,
                          warp=warp_record,
                          shape=list(current[i].shape), dtype=str(current.dtype))
            folder = self.root / f'slot_{i:02d}' / safe_name(meta['scene_token'])
            path = folder / (f"step_{self.step:04d}_seq_{meta['sequence_instance']:04d}_"
                             f"frame_{position:03d}_{safe_name(meta['token'])}.png")
            record['color_limits'] = draw_triplet(
                path, current_heat, history_heat, aligned_heat, pixel_mask, record,
                self.model.bev_warp.point_cloud_range)
            record['image'] = str(path.relative_to(self.root))
            self.pending.append(record)

    def after_forward(self):
        cache = self.model.bev_cache
        if len(self.pending) != cache.bev.shape[0]:
            raise AssertionError('Missing captured BEV slots')
        if cache.bev.requires_grad or cache.bev.grad_fn is not None:
            raise AssertionError('Updated cache retains a computation graph')
        for record in self.pending:
            slot = record['local_slot']
            digest, _ = signature(cache.bev[slot])
            #* (11.12) 融合启用时缓存应匹配融合输出，而不是融合前的当前 BEV。
            if digest != record['cached_sha256'] or cache.metas[slot] != record['current']:
                raise AssertionError(f'Slot {slot}: cache update does not match current BEV')
            record['cache_update_matches'] = True
            self.previous[slot] = record
            self.records.append(record)
        self.pending = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='本阶段的 stream_cache YAML 配置')
    parser.add_argument('checkpoint', help='结构匹配的原 DAOcc 或流式 DAOcc 权重')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--steps', type=int, default=16)
    #* (11.12) 可单独检查 train 或 val，默认保留两者均执行的行为。
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both',
                        help='选择处理的数据集，默认 both；只重建所选侧的输出子目录')
    parser.add_argument('--batch-size', type=int, default=4, help='训练侧每卡 BS；验证侧每卡固定为 1')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=int, default=0, help='CUDA_VISIBLE_DEVICES 内的逻辑 GPU 编号')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=None, help='默认沿用配置中的 seed')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    args = parser.parse_args()
    splits = ('train', 'val') if args.split == 'both' else (args.split,)
    if args.steps < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('steps/batch-size 必须为正数，workers 必须非负')
    #* (11.12) torchrun 每卡一进程，复用正式分布式 sampler；单卡命令保持兼容。
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.distributed = args.world_size > 1
    args.rank = int(os.environ.get('RANK', '0'))
    if args.distributed:
        args.device = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(args.device)
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
    root = Path(args.out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_root = root
    if args.distributed:
        root = root / f'rank_{args.rank:02d}'
        if root.is_symlink():
            parser.error(f'拒绝使用符号链接 rank 目录：{root}')
        root.mkdir(parents=True, exist_ok=True)
    raw = TorchpackConfig()
    raw.load(str(Path(args.config).expanduser().resolve()), recursive=True)
    cfg = Config(recursive_eval(raw))
    cfg.merge_from_dict(args.cfg_options)
    if not cfg.get('streaming', {}).get('enabled', False):
        parser.error('请提供 streaming.enabled=true 的独立流式训练配置')
    cfg = prepare_streaming_config(cfg)
    if cfg.model.get('bev_warp') is None:
        parser.error('对齐可视化需要 streaming.bev_warp.enabled=true')
    cfg.data.samples_per_gpu, cfg.data.workers_per_gpu = args.batch_size, args.workers
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.model.cache_enabled = True
    # 完整 checkpoint 提供参数；不额外读取 ResNet 分类预训练权重。
    camera = cfg.model.encoders.get('camera')
    if camera:
        camera.backbone.init_cfg = None
        camera.backbone.pop('pretrained', None)
    torch.cuda.set_device(args.device)
    set_random_seed(cfg.seed, deterministic=cfg.get('deterministic', False))
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)
    #* (11.12) 两套数据与真实缓存分别检查；每套重新加载 checkpoint，隔离训练态 BN 变化。
    # 只重建本次选中的子目录；summary.json 记录本次运行，未选中的结果目录保留。
    protected = [Path(__file__).resolve().parents[1], Path.home().resolve(),
                 Path(args.config).expanduser().resolve(), Path(args.checkpoint).expanduser().resolve()]
    targets = [root / split for split in splits]
    for target in targets:
        if target.is_symlink():
            parser.error(f'拒绝清理符号链接输出目录：{target}')
        resolved = target.resolve()
        if any(resolved == item or resolved in item.parents for item in protected):
            parser.error(f'输出目录包含代码、配置、主目录或权重，拒绝清理：{target}')
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    statuses = {}
    for split in splits:
        try:
            statuses[split] = run_split(cfg, args, root / split, split)
        except Exception as exc:
            # 多卡失败立即退出，让 torchrun 终止其余进程，避免卡在同步 BN/汇总。
            if args.distributed:
                raise
            statuses[split] = f'failed: {type(exc).__name__}: {exc}'
            print(f'{split} 检查失败，已记录错误。', file=sys.stderr)
            traceback.print_exc()
            manifest = root / split / 'manifest.json'
            if not manifest.exists():
                manifest.write_text(json.dumps(dict(split=split, status='failed',
                    error=statuses[split]), ensure_ascii=False, indent=2) + '\n')
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    (root / 'summary.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + '\n')
    if args.distributed:
        reports = [None] * args.world_size
        dist.all_gather_object(reports, dict(rank=args.rank, splits=statuses))
        if args.rank == 0:
            (output_root / 'summary.json').write_text(
                json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
        dist.destroy_process_group()
    print('本检查覆盖采样与前向缓存行为，不执行反向传播、优化器更新或完整验证集指标计算。')
    if any(value.startswith('failed:') for value in statuses.values()):
        raise SystemExit(1)


def run_split(cfg, args, root, split):
    set_random_seed(cfg.seed, deterministic=cfg.get('deterministic', False))
    batch_size = args.batch_size if split == 'train' else 1
    if split == 'train':
        dataset = build_dataset(cfg.data.train)
        loader = build_stream_dataloader(dataset, cfg, distributed=args.distributed)
    else:
        loader = build_stream_val_dataloader(cfg, distributed=args.distributed)
        dataset = loader.dataset
        if args.steps > len(loader):
            raise ValueError(f'val 仅有 {len(loader)} 步，不能执行 {args.steps} 步；不会循环重复验证帧')
    # 汇总完整验证分片索引，检查各卡无重复、无遗漏；无需执行完整验证前向。
    if args.distributed and split == 'val':
        assignments = [None] * args.world_size
        dist.all_gather_object(assignments, [r['idx'] for r in loader.sampler.requests])
        flat = [index for indices in assignments for index in indices]
        if len(flat) != len(dataset) or len(set(flat)) != len(dataset):
            raise AssertionError('Validation rank partitions overlap or omit frames')
        owners = {}
        for rank, indices in enumerate(assignments):
            for index in indices:
                scene = dataset.data_infos[index]['scene_token']
                if scene in owners and owners[scene] != rank:
                    raise AssertionError(f'Validation scene {scene} is split across ranks')
                owners[scene] = rank
    model = build_model(deepcopy(cfg.model))
    if cfg.get('sync_bn'):
        sync_cfg = cfg.sync_bn if isinstance(cfg.sync_bn, dict) else {}
        model = convert_sync_batchnorm(model, exclude=sync_cfg.get('exclude', []))
    load_checkpoint(model, str(Path(args.checkpoint).expanduser()), map_location='cpu', strict=True)
    if args.precision == 'fp16':
        wrap_fp16_model(model)
    model.cuda(args.device).train(split == 'train')
    model.reset_stream()
    observer = CacheObserver(model, dataset, root, slot_offset=args.rank * batch_size)
    if split == 'val':
        # 验证边界以正式验证 sampler 为准，不以训练 scene_token 分组替代。
        groups = {}
        for request in loader.sampler.requests:
            groups.setdefault(request['sequence_instance'], []).append(request['idx'])
        observer.lookup = {index: (group, position, len(indices))
                           for group, indices in groups.items()
                           for position, index in enumerate(indices)}
    original = model.process_history
    #* (11.12) 跳空帧时允许帧位置及全空子序列编号有间隔；请求仍与 sampler 逐项核对。
    model._skip_empty_gt_visual_check = bool(getattr(dataset, 'skip_empty_gt', False) and split == 'train')
    model._temporal_skip_visual_check = bool(getattr(dataset, 'skip_prob', 0) > 0 and split == 'train')

    def capture(this, current, history, valid):
        observer.capture(current, history, valid)
        #* (11.12) 捕获实际融合输出，供本帧写缓存及下一帧读缓存校验。
        output = original(current, history, valid)
        for record in observer.pending:
            record['cached_sha256'] = signature(output[record['local_slot']])[0]
            record['cache_source'] = 'fused' if this.history_fusion is not None else 'current'
        return output

    model.process_history = MethodType(capture, model)
    report = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
                  split=split, rank=args.rank, world_size=args.world_size,
                  batch_size=batch_size, requested_steps=args.steps, seed=cfg.seed,
                  precision=args.precision, sequences_split_num=dataset.sequences_split_num,
                  augmentation_mode=observer.augmentation_mode,
                  augmentation_scope='image/LiDAR geometry; GridMask and point shuffle remain independent',
                  sequence_groups=(len(groups) if split == 'val' else len(dataset.sequence_indices)),
                  completed_steps=0,
                  mode=('training forward; no backward/optimizer' if split == 'train'
                        else 'eval forward; real validation_stream cache; no metrics'),
                  feature='decoder.neck BEV; raw history / warped history / current; mean(abs(channels))',
                  alignment=dict(point_cloud_range=list(model.bev_warp.point_cloud_range),
                                 plane_z=model.bev_warp.plane_z, tensor_layout='B,C,X,Y',
                                 grid_sample_order='Y,X', align_corners=True,
                                 grid_reference='cell centers',
                                 transform='A_prev @ inv(G_prev) @ G_cur @ inv(A_cur)',
                                 geometry='G=ego2global@lidar2ego; planar ego-motion only'),
                  status='running', records=observer.records)
    def save_report():
        report['subsequence_transitions_per_slot'] = {
            str(i): observer.transitions.get(i, 0) for i in range(batch_size)}
        report['slots_without_transition'] = [i for i in range(batch_size)
                                               if observer.transitions.get(i, 0) == 0]
        (root / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    try:
        with (model.validation_stream() if split == 'val' else nullcontext()):
            iterator = iter(loader)
            expected_requests = iter(loader.batch_sampler if split == 'train' else loader.sampler)
            for step in range(args.steps):
                observer.step = step
                data = scatter(next(iterator), [args.device])[0]
                expected = next(expected_requests)
                expected = expected if split == 'train' else [expected]
                for meta, request in zip(data['stream_metas'], expected):
                    if (meta['dataset_index'] != request['idx'] or any(
                            meta[key] != request[key] for key in
                            ('slot', 'sequence_instance', 'sequence_start'))):
                        raise AssertionError('Loaded frame does not match this rank sampler request')
                    if meta['aug_config'] != request.get('aug_config'):
                        raise AssertionError('Loaded augmentation does not match this rank sampler request')
                # 位姿/增强缓存必须与本次模型输入一致，防止 pipeline 之后的记录失配。
                for local_slot, meta in enumerate(data['stream_metas']):
                    for key in ('lidar2ego', 'lidar_aug_matrix', 'img_aug_matrix', 'occ_aug_matrix'):
                        if key not in meta:
                            continue
                        actual_matrix = data[key][local_slot].detach().double().cpu().numpy()
                        if not np.allclose(actual_matrix, np.asarray(meta[key]), atol=1e-6, rtol=1e-6):
                            raise AssertionError(f'Slot {local_slot}: {key} differs from model input')
                with torch.no_grad():
                    outputs = model(**data)
                observer.after_forward()
                report['completed_steps'] = step + 1
                save_report()
                print(f'rank {args.rank} {split} step {step + 1}/{args.steps}: saved {batch_size} slots; '
                      f'history valid={[r["history_valid"] for r in observer.records[-batch_size:]]}', flush=True)
                del outputs, data
        report['status'] = 'passed_for_observed_steps'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['pending_records'] = observer.pending
        raise
    finally:
        save_report()
        # 删除实例覆盖，恢复类方法，避免 bound method 自引用滞留显存。
        del model.process_history
        model.reset_stream()
    print(f'结果：{root}')
    if report['slots_without_transition']:
        print(f'{split} 以下槽位尚未观察到序列切换，可增加 --steps 扩大观察范围：',
              report['slots_without_transition'])
    return report['status']


if __name__ == '__main__':
    main()
