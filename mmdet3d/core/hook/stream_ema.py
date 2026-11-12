"""Iteration EMA snapshots for streaming training; never swap evaluation weights."""
import math
from pathlib import Path

import torch
from mmcv.runner import Hook, get_dist_info


class StreamEMAHook(Hook):
    def __init__(self, decay=0.9990, interval=2000, save_last=True,
                 max_keep_ckpts=1, resume=None):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay = decay
        self.interval = interval
        self.save_last = save_last
        self.max_keep_ckpts = max_keep_ckpts
        self.resume = resume
        self.updates = 0
        self.state = None

    def before_run(self, runner):
        if get_dist_info()[0] != 0:
            return
        model = runner.model.module
        # Only parameters and persistent buffers: no BEV caches or DDP/SyncBN copies.
        self.state = {key: (value.detach().float().clone() if value.is_floating_point()
                            else value.detach().clone())
                      for key, value in model.state_dict().items()}
        #* (11.12) 与 StreamOcc 一致：EMA 断点仅由 resume_from_EMA 显式指定。
        # 不根据普通 resume_from 自动搜索或推断 EMA 文件；未指定时从当前主模型初始化。
        if self.resume:
            saved = torch.load(self.resume, map_location='cpu')
            weights = saved['state_dict']
            if weights.keys() != self.state.keys():
                raise ValueError('EMA snapshot keys do not match model')
            for key, target in self.state.items():
                if weights[key].shape != target.shape:
                    raise ValueError(f'EMA snapshot shape mismatch: {key}')
                target.copy_(weights[key])
            self.updates = int(saved['updates'])
            runner.logger.info('Restored streaming EMA from %s (%d updates)',
                               self.resume, self.updates)
        runner.logger.info('Streaming EMA enabled on rank 0; evaluation uses current training weights.')

    @torch.no_grad()
    def after_train_iter(self, runner):
        if self.state is None:
            return
        # Same per-iteration update and exponential warmup as the original ModelEMA.
        # An AMP-skipped optimizer step still counts as an iteration, as in that hook.
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / 2000))
        for key, value in runner.model.module.state_dict().items():
            if self.state[key].is_floating_point():
                self.state[key].mul_(decay).add_(value.detach().to(self.state[key].dtype), alpha=1 - decay)
            else:
                self.state[key].copy_(value.detach())
        if (self.every_n_iters(runner, self.interval) or
                (self.save_last and self.is_last_iter(runner))):
            self.save_checkpoint(runner)

    def save_checkpoint(self, runner):
        directory = Path(runner.work_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'iter_{runner.iter + 1}_ema.pth'
        temporary = destination.with_suffix('.pth.tmp')
        torch.save(dict(state_dict={key: value.cpu() for key, value in self.state.items()},
                        updates=self.updates, iter=runner.iter + 1,
                        meta=dict(iter=runner.iter + 1, ema=True, decay=self.decay)), temporary)
        temporary.replace(destination)
        runner.logger.info('Saved streaming EMA checkpoint: %s', destination)
        if self.max_keep_ckpts > 0:
            snapshots = [path for path in directory.glob('iter_*_ema.pth')
                         if path.name.split('_')[1].isdigit()]
            snapshots.sort(key=lambda path: int(path.name.split('_')[1]))
            for path in snapshots[:-self.max_keep_ckpts]:
                path.unlink()
