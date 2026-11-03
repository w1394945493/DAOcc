"""仅供新增轮内续训模式使用；原epoch保存、EMA和评估Hook保持不变。"""
from pathlib import Path

from mmcv.runner import HOOKS, Hook
from mmdet.core import EvalHook, DistEvalHook
from .ema import MEGVIIEMAHook


@HOOKS.register_module()
class ResumableEMAHook(MEGVIIEMAHook):
    def before_run(self, runner):
        if self.resume is not None:
            raise ValueError('新模式EMA随普通checkpoint恢复，不接受独立EMA_RESUME')
        super().before_run(runner)
        state = runner._resume_ema
        if state is not None:
            runner.ema_model.ema.load_state_dict(state['state_dict'], strict=True)
            runner.ema_model.updates = state['updates']
            runner._resume_ema = None
        elif runner._resume_state is not None:
            raise ValueError('续训checkpoint缺少EMA状态')

    def after_train_epoch(self, runner):
        # 普通checkpoint内嵌EMA，不额外按epoch写文件。
        pass


@HOOKS.register_module()
class ResumableCheckpointHook(Hook):
    def __init__(self, interval=2000, by_epoch=False, max_keep_ckpts=1,
                 save_last=True, save_optimizer=True):
        if by_epoch or interval <= 0 or not save_optimizer:
            raise ValueError('新保存Hook要求by_epoch=False、正interval和save_optimizer=True')
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last

    def before_run(self, runner):
        if type(runner).__name__ != 'ResumableEpochBasedRunner':
            raise ValueError('ResumableCheckpointHook仅用于新Runner')
        # EMA必须在保存之前更新；MMCV把checkpoint固定注册在NORMAL优先级。
        for hook in runner.hooks:
            if isinstance(hook, ResumableEMAHook) and hook.priority >= self.priority:
                raise ValueError('ResumableEMAHook需设置priority: ABOVE_NORMAL')

    def after_train_iter(self, runner):
        step = runner.iter + 1
        if step % self.interval and not (self.save_last and step == runner.max_iters):
            return
        if runner.rank != 0:
            return
        path = Path(runner.work_dir) / f'iter_{step}.pth'
        runner.meta.setdefault('hook_msgs', {})['last_ckpt'] = str(path)
        runner.save_checkpoint(runner.work_dir)
        runner.logger.info('保存迭代checkpoint（内含EMA/采样进度）：%s', path)
        files = [p for p in Path(runner.work_dir).glob('iter_*.pth')
                 if p.stem[5:].isdigit()]
        files.sort(key=lambda p: int(p.stem[5:]))
        if self.max_keep_ckpts > 0:
            for old in files[:-self.max_keep_ckpts]:
                old.unlink()


class IterationEvalMixin:
    def __init__(self, *args, evaluate_last=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_last = evaluate_last

    def _should_evaluate(self, runner):
        return (super()._should_evaluate(runner) or
                (not self.by_epoch and self.evaluate_last and
                 runner.iter + 1 == runner.max_iters and runner._in_train_iter))

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        # 恢复每个子模块原先的training状态，保留冻结BN等特殊设置。
        modes = [(module, module.training) for module in runner.model.modules()]
        try:
            super()._do_evaluate(runner)
        finally:
            for module, mode in modes:
                module.training = mode
        # rank0完成数据集评估前，其余rank不要抢先进入下一步DDP训练。
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


class IterationEvalHook(IterationEvalMixin, EvalHook):
    pass


class IterationDistEvalHook(IterationEvalMixin, DistEvalHook):
    pass
