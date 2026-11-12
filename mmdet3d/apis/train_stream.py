"""Opt-in sequence training. Existing DAOcc training paths are unchanged."""
from copy import deepcopy
from functools import partial

from mmcv.parallel import collate, MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (build_optimizer, build_runner, get_dist_info,
                         Fp16OptimizerHook, OptimizerHook)
from torch.utils.data import DataLoader
import torch
from mmdet.datasets.builder import worker_init_fn
from mmdet3d.utils import get_root_logger


def prepare_streaming_model_config(cfg):
    """Configure the streaming model without requiring training data or a runner."""
    from mmdet3d.models.fusion_models.streaming_bevfusion import StreamingBEVFusion
    if cfg.model.type not in ('BEVFusion', 'StreamingBEVFusion'):
        raise ValueError('Streaming adapter requires the DAOcc BEVFusion model')
    cfg.model.type = 'StreamingBEVFusion'
    cfg.model.cache_enabled = cfg.streaming.get('cache_enabled', True)
    #* (11.12) 流式 BEV 对齐使用源 LiDAR BEV 范围，与占据输出的范围分开配置。
    warp_cfg = dict(cfg.streaming.get('bev_warp', {}))
    if warp_cfg.pop('enabled', False):
        warp_cfg.setdefault('point_cloud_range', deepcopy(cfg.point_cloud_range))
        cfg.model.bev_warp = warp_cfg
    else:
        cfg.model.bev_warp = None
    #* (11.12) 可选 StreamOcc 风格历史修正与拼接融合，旧配置默认关闭。
    fusion_cfg = dict(cfg.streaming.get('history_fusion', {}))
    cfg.model.history_fusion = fusion_cfg if fusion_cfg.pop('enabled', False) else None
    return cfg


def prepare_streaming_config(cfg):
    """Called before model/dataset construction, only by opted-in training entries."""
    if not cfg.get('streaming', {}).get('enabled', False):
        return cfg
    # Lazy registration: original configurations do not import stream modules.
    from mmdet3d.datasets.nuscenes_sequence_dataset import NuScenesSequenceDataset
    from mmdet3d.models.fusion_models.streaming_bevfusion import StreamingBEVFusion
    train_cfg = deepcopy(cfg.data.train)
    if train_cfg.get('type') == 'CBGSDataset':
        train_cfg = deepcopy(train_cfg['dataset'])
    if train_cfg.get('type') not in ('NuScenesDatasetOccupancy', 'NuScenesSequenceDataset'):
        raise ValueError('Streaming requires an unwrapped nuScenes occupancy dataset')
    train_cfg.update(type='NuScenesSequenceDataset', filter_empty_gt=False, resample=False)
    cfg.data.train = train_cfg
    #* (11.12) 训练与独立评估共用流式模型配置。
    prepare_streaming_model_config(cfg)
    if cfg.runner.type != 'IterBasedRunner':
        raise ValueError('Streaming requires runner.type=IterBasedRunner')
    # Torchpack recursively merges default.yaml, including its max_epochs key.
    cfg.runner.pop('max_epochs', None)
    if not isinstance(cfg.runner.get('max_iters'), int) or cfg.runner.max_iters <= 0:
        raise ValueError('Set a positive runner.max_iters for streaming')
    if cfg.seed is None:
        raise ValueError('Streaming requires an explicit shared integer seed')
    if cfg.get('custom_hooks'):
        raise ValueError('Initial stream configuration requires custom_hooks: null; epoch EMA/resume hooks are incompatible')
    return cfg


def build_stream_dataloader(dataset, cfg, distributed):
    from mmdet3d.datasets.samplers import GroupInBatchSampler
    rank, world_size = get_dist_info()
    if not distributed and world_size != 1:
        raise ValueError('Initialized process group requires distributed=True')
    batch_size = cfg.data.samples_per_gpu
    workers = cfg.data.workers_per_gpu
    #*=============================================================================#
    #* GroupInBatchSampler
    sampler = GroupInBatchSampler(dataset, batch_size, rank, world_size, cfg.seed)
    dataset.set_epoch(0)  # Infinite stream has no implicit epoch transitions.
    return DataLoader(
        dataset, 
        batch_sampler=sampler, 
        num_workers=workers,
        collate_fn=partial(collate, samples_per_gpu=batch_size),
        worker_init_fn=partial(worker_init_fn, num_workers=workers, rank=rank, seed=cfg.seed),
        pin_memory=False,
    )


def build_stream_val_dataloader(cfg, distributed, split="val"):
    """Training-time validation, same BS=1 contract as StreamOcc."""
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.nuscenes_sequence_dataset import NuScenesSequenceDataset
    from mmdet3d.datasets.samplers import SceneDistributedSampler
    #* (11.12) 独立评估复用同一序列加载器，但读取 data.test。
    val_cfg = deepcopy(cfg.data[split])
    batch_size = val_cfg.pop('samples_per_gpu', 1)
    if batch_size != 1:
        raise ValueError('StreamOcc-style validation requires samples_per_gpu=1')
    if val_cfg.get('sequences_split_num', 1) != 1:
        raise ValueError('Validation uses full sequences, not training subsequence splitting')
    val_cfg.update(type='NuScenesSequenceDataset', test_mode=True, sequences_split_num=1)
    dataset = build_dataset(val_cfg)
    rank, world_size = get_dist_info()
    if not distributed and world_size != 1:
        raise ValueError('Initialized process group requires distributed=True')
    workers = cfg.data.workers_per_gpu
    sampler = SceneDistributedSampler(dataset, rank, world_size)
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=workers,
                      collate_fn=partial(collate, samples_per_gpu=1), pin_memory=False,
                      worker_init_fn=partial(worker_init_fn, num_workers=workers,
                                             rank=rank, seed=cfg.seed))


def train_stream_model(model, dataset, cfg, distributed=False, validate=False, timestamp=None):
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.nuscenes_sequence_dataset import NuScenesSequenceDataset
    from mmdet3d.models.fusion_models.streaming_bevfusion import StreamingBEVFusion
    from mmdet3d.core.hook.stream_eval_hooks import StreamEvalHook, StreamDistEvalHook

    logger = get_root_logger()
    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if len(datasets) != 1 or not isinstance(datasets[0], NuScenesSequenceDataset):
        raise ValueError('Use prepare_streaming_config before constructing the sequence dataset')
    if not isinstance(model, StreamingBEVFusion):
        raise ValueError('Streaming training requires StreamingBEVFusion')
    if cfg.runner.type != 'IterBasedRunner' or cfg.get('custom_hooks'):
        raise ValueError('Use the independent IterBasedRunner stream configuration')
    if cfg.lr_config.get('by_epoch', True) or cfg.checkpoint_config.get('by_epoch', True):
        raise ValueError('Stream LR and checkpoints must use by_epoch: false')
    if cfg.get('debug', False):
        raise ValueError('Epoch-style debug freezing is not supported by the stream runner')
    
    #*=============================================================#
    #* dataloader
    loader = build_stream_dataloader(datasets[0], cfg, distributed)
    logger.info('Sequence training: %d sequence groups, %d frames; history is cached but NOT fused. '
                'Streaming validation uses current weights and a separate cache.',
                len(datasets[0].sequence_indices), len(datasets[0]))
    #* (11.12) 明确记录当前流式训练的几何增强粒度。
    logger.info('keep_consistent_seq_aug=%s: %s geometric augmentation.',
                datasets[0].keep_consistent_seq_aug,
                'per-subsequence' if datasets[0].keep_consistent_seq_aug else 'per-frame')
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model = MMDataParallel(model.cuda(), device_ids=[0])
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(cfg.runner, default_args=dict(
        model=model, optimizer=optimizer, work_dir=cfg.run_dir, logger=logger, meta={}))
    runner.timestamp = timestamp
    optimizer_cfg = dict(cfg.optimizer_config)
    if 'cumulative_iters' in optimizer_cfg or 'Cumulative' in optimizer_cfg.get('type', ''):
        raise ValueError('Initial streaming implementation does not support gradient accumulation')
    fp16_cfg = cfg.get('fp16')
    if fp16_cfg is not None:
        optimizer_cfg.pop('type', None)
        optimizer_hook = Fp16OptimizerHook(**optimizer_cfg, **fp16_cfg, distributed=distributed)
    else:
        optimizer_hook = optimizer_cfg if 'type' in optimizer_cfg else OptimizerHook(**optimizer_cfg)
    runner.register_training_hooks(
        cfg.lr_config, optimizer_hook, cfg.checkpoint_config, cfg.log_config,
        cfg.get('momentum_config'))
    #* (11.12) 迭代 EMA 在优化器之后、普通 checkpoint 之前更新；不替换训练/评估权重。
    # 使用独立 state_dict，避免复制历史缓存及分布式包装对象。
    ema_cfg = dict(cfg.streaming.get('ema', {}))
    if ema_cfg.pop('enabled', False):
        from mmdet3d.core.hook.stream_ema import StreamEMAHook
        ema_cfg.setdefault('interval', cfg.checkpoint_config.get('interval', 2000))
        ema_cfg.setdefault('save_last', cfg.checkpoint_config.get('save_last', True))
        ema_cfg.setdefault('max_keep_ckpts', cfg.checkpoint_config.get('max_keep_ckpts', 1))
        #* (11.12) 参照 StreamOcc，普通权重与 EMA 权重分别由 resume_from、
        # resume_from_EMA 恢复；评估 Hook 仍使用当前训练模型。
        ema_cfg['resume'] = cfg.get('resume_from_EMA')
        runner.register_hook(StreamEMAHook(**ema_cfg), priority=49)
    if validate:
        #* (11.12) 训练中验证按 StreamOcc 使用每卡一条连续流，而非普通交错采样。
        val_loader = build_stream_val_dataloader(cfg, distributed)
        eval_cfg = dict(cfg.get('evaluation', {}))
        eval_cfg['by_epoch'] = False
        hook = StreamDistEvalHook if distributed else StreamEvalHook
        runner.register_hook(hook(val_loader, **eval_cfg), priority='LOW')
    if cfg.get('resume_from'):
        runner.resume(cfg.resume_from)
        logger.info('Resumed weights/optimizer/iteration; sampler restarts from seed and BEV history is empty. '
                    'This is not exact sample-position recovery.')
    elif cfg.get('load_from'):
        runner.load_checkpoint(cfg.load_from)
    model.module.reset_stream()
    try:
        runner.run([loader], [('train', 1)])
    finally:
        model.module.reset_stream()
