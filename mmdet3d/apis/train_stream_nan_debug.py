"""流式训练的独立 NaN 诊断入口，不修改正常流式训练路径。"""

from .train_stream import (
    build_stream_dataloader,
    build_stream_val_dataloader,
)

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
import torch

from mmdet3d.utils import get_root_logger
from mmdet3d.utils.nan_debug_persistent import install_persistent_nan_debug


def train_stream_model_nan_debug(
        model, dataset, cfg, distributed=False, validate=False, timestamp=None):
    """保持原流式训练流程，仅额外安装持久化 NaN 诊断 Hook。"""
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

    loader = build_stream_dataloader(datasets[0], cfg, distributed)
    logger.info(
        'Sequence NaN-debug training: %d sequence groups, %d frames.',
        len(datasets[0].sequence_indices), len(datasets[0]))
    logger.info(
        'keep_consistent_seq_aug=%s: %s geometric augmentation.',
        datasets[0].keep_consistent_seq_aug,
        'per-subsequence' if datasets[0].keep_consistent_seq_aug else 'per-frame')

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model = MMDataParallel(model.cuda(), device_ids=[0])

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(cfg.runner, default_args=dict(
        model=model, optimizer=optimizer, work_dir=cfg.run_dir,
        logger=logger, meta={}))
    runner.timestamp = timestamp

    optimizer_cfg = dict(cfg.optimizer_config)
    if ('cumulative_iters' in optimizer_cfg
            or 'Cumulative' in optimizer_cfg.get('type', '')):
        raise ValueError('NaN debug does not support gradient accumulation')
    fp16_cfg = cfg.get('fp16')
    if fp16_cfg is not None:
        optimizer_cfg.pop('type', None)
        optimizer_hook = Fp16OptimizerHook(
            **optimizer_cfg, **fp16_cfg, distributed=distributed)
    else:
        optimizer_hook = (optimizer_cfg if 'type' in optimizer_cfg
                          else OptimizerHook(**optimizer_cfg))
    runner.register_training_hooks(
        cfg.lr_config, optimizer_hook, cfg.checkpoint_config,
        cfg.log_config, cfg.get('momentum_config'))

    #* (11.25) 诊断 Hook 紧邻 OptimizerHook 前后执行，可区分前向损失异常、
    #* 反向梯度异常和参数/优化器状态污染；正常训练入口不受影响。
    nan_debug_cfg = dict(cfg.get('nan_debug', {}))
    nan_debug_cfg.setdefault('enabled', True)
    install_persistent_nan_debug(runner, nan_debug_cfg)

    # 与原流式入口保持一致：EMA、验证、恢复及缓存生命周期均不改变。
    ema_cfg = dict(cfg.streaming.get('ema', {}))
    if ema_cfg.pop('enabled', False):
        from mmdet3d.core.hook.stream_ema import StreamEMAHook
        ema_cfg.setdefault('interval', cfg.checkpoint_config.get('interval', 2000))
        ema_cfg.setdefault('save_last', cfg.checkpoint_config.get('save_last', True))
        ema_cfg.setdefault('max_keep_ckpts', cfg.checkpoint_config.get('max_keep_ckpts', 1))
        ema_cfg['resume'] = cfg.get('resume_from_EMA')
        runner.register_hook(StreamEMAHook(**ema_cfg), priority=49)

    if validate:
        val_loader = build_stream_val_dataloader(cfg, distributed)
        eval_cfg = dict(cfg.get('evaluation', {}))
        eval_cfg['by_epoch'] = False
        hook = StreamDistEvalHook if distributed else StreamEvalHook
        runner.register_hook(hook(val_loader, **eval_cfg), priority='LOW')

    if cfg.get('resume_from'):
        runner.resume(cfg.resume_from)
        logger.info(
            'Resumed weights/optimizer/iteration; sampler restarts from seed '
            'and BEV history is empty. This is not exact sample-position recovery.')
    elif cfg.get('load_from'):
        runner.load_checkpoint(cfg.load_from)

    model.module.reset_stream()
    try:
        runner.run([loader], [('train', 1)])
    finally:
        model.module.reset_stream()
