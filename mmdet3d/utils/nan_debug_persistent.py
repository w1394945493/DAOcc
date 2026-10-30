"""独立训练入口使用的持续 NaN/Inf 诊断模块。

启用参数：--nan_debug.enabled True
可选参数：--nan_debug.out_dir /path/to/output
          --nan_debug.save_batch True
          --nan_debug.stop_after_consecutive_nan 10

默认每次异常都记录，不限制文件数量；持续异常可能产生大量 batch 文件。
总 loss 连续 ``stop_after_consecutive_nan`` 步异常后，所有 rank 协同停止。
关闭 enabled 后，本模块不会被导入，亦无张量扫描开销。
本模块只诊断训练，不接管优化器的跳步逻辑，不保证异常后训练可恢复。
"""
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import DataContainer
from mmcv.runner import Hook, OptimizerHook


def _cpu(value):
    """解开 DataContainer，保存实际增强后数据，禁止保留 GPU/计算图引用。

    三维框存为带类型说明的普通字典；batch.pt 是诊断快照，不是可直接
    传给模型的 batch，重放时需重新构建框对象及 DataContainer。
    """
    if isinstance(value, DataContainer):
        return _cpu(value.data)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy()) if value.dtype.kind in 'biufc' else value.tolist()
    if isinstance(value, dict):
        return {str(k): _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    if hasattr(value, 'tensor') and torch.is_tensor(value.tensor):
        return {'type': type(value).__name__, 'tensor': _cpu(value.tensor)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sample_metas(value):
    """collate 后 metas 可能有设备分组的嵌套列表，按本 rank 顺序展开。"""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [meta for group in value for meta in _sample_metas(group)]
    return []


def _summary(value):
    """JSON 只保留张量统计；min/max 仅对有限值计算。"""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        finite = torch.isfinite(value)
        result = dict(shape=list(value.shape), dtype=str(value.dtype),
                      nonfinite=int((~finite).sum()))
        if value.is_floating_point():
            result.update(nan=int(torch.isnan(value).sum()),
                          inf=int(torch.isinf(value).sum()))
        if finite.any() and not value.is_complex():
            vals = value[finite]
            result.update(min=vals.min().item(), max=vals.max().item())
        return result
    if isinstance(value, dict):
        return {k: _summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summary(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # 使用合法 JSON，不写非标准 NaN 常量。
    return value


def _bad_tensors(items, max_details=None):
    """批量检查有限性，并限制写入报告的异常张量明细数量。"""
    groups = {}
    for name, tensor in items:
        if torch.is_tensor(tensor) and (tensor.is_floating_point() or tensor.is_complex()):
            groups.setdefault(tensor.device, []).append((name, tensor.detach()))
    bad = {}
    total_bad = 0
    for group in groups.values():
        flags = torch.stack([torch.isfinite(t).all() for _, t in group]).cpu().tolist()
        for (name, tensor), good in zip(group, flags):
            if not good:
                total_bad += 1
                if max_details is None or len(bad) < max_details:
                    bad[name] = _summary(tensor)
    if max_details is not None and total_bad > len(bad):
        bad['__omitted_problem_tensors__'] = {
            'total_bad_tensors': total_bad,
            'reported_bad_tensors': len(bad),
            'omitted_bad_tensors': total_bad - len(bad),
        }
    return bad


class _Recorder:
    def __init__(self, runner, config, optimizer_hook):
        self.config = dict(config)
        configured_root = config.get('out_dir')
        self.root = Path(configured_root or (Path(runner.work_dir) / 'nan_debug'))
        # 分开不同启动，避免同一个输出目录重跑后覆盖证据。
        self.root = self.root / str(runner.timestamp)
        self.optimizer_hook = optimizer_hook
        self.saved_batch = False
        self.scale_before = None
        self.consecutive_bad_losses = 0
        self.stop_after = int(config.get('stop_after_consecutive_nan', 10))
        self.max_problem_tensors = int(config.get('max_problem_tensors', 50))
        self.checkpoint_interval = int(config.get('rolling_checkpoint_interval', 0))
        self.checkpoint_keep = int(config.get('rolling_checkpoint_keep', 3))
        self.activation_tracker = None
        self.saved_activation_history = False
        if self.stop_after <= 0:
            raise ValueError('nan_debug.stop_after_consecutive_nan must be positive')
        if self.max_problem_tensors <= 0:
            raise ValueError('nan_debug.max_problem_tensors must be positive')
        if self.checkpoint_interval < 0 or self.checkpoint_keep <= 0:
            raise ValueError('rolling checkpoint interval must be >= 0 and keep must be > 0')

    @staticmethod
    def _sync(runner, value):
        """所有 rank 在固定位置调用；禁止仅在异常分支内执行 collective。"""
        if not (dist.is_available() and dist.is_initialized()):
            return int(value)
        device = next(runner.model.parameters()).device if dist.get_backend() == 'nccl' else 'cpu'
        flag = torch.tensor(int(value), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return int(flag.item())

    def scale(self):
        scaler = getattr(self.optimizer_hook, 'loss_scaler', None)
        if scaler is None:
            return None
        if hasattr(scaler, 'get_scale'):
            return float(scaler.get_scale())
        return float(scaler.loss_scale) if hasattr(scaler, 'loss_scale') else None

    def record(self, runner, stage, problems, globally_abnormal=None):
        # 即使本 rank 正常，其他 rank 异常时也保存本地 batch 和正常状态。
        global_flag = (self._sync(runner, bool(problems))
                       if globally_abnormal is None else int(globally_abnormal))
        if not global_flag:
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        folder = self.root / ('epoch_%03d_iter_%08d' % (runner.epoch + 1, runner.iter + 1)) / ('rank_%03d' % rank)
        error = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            batch = _cpu(runner.data_batch)
            samples = _sample_metas(batch.get('metas', []))
            if self.config.get('save_batch', True) and not self.saved_batch:
                torch.save(batch, str(folder / 'batch.pt'))
                self.saved_batch = True
            # 这些文件均为本 rank 的记录；log_vars 可能已经过跨卡平均。
            outputs = getattr(runner, 'outputs', {}) if stage != 'before_forward' else {}
            report = dict(epoch=runner.epoch + 1, iteration=runner.iter + 1,
                          inner_iteration=runner.inner_iter + 1, rank=rank,
                          stage=stage, local_abnormal=bool(problems), problems=problems,
                          local_batch_size=len(samples),
                          samples=[dict(batch_index=i, meta=_summary(meta))
                                   for i, meta in enumerate(samples)],
                          scale_before=self.scale_before, scale_current=self.scale(),
                          lr=runner.current_lr(), batch=_summary(batch),
                          local_total_loss=_summary(_cpu(outputs.get('loss'))),
                          reduced_log_vars=_summary(_cpu(outputs.get('log_vars', {}))),
                          note='Each rank stores its local batch; gradients may be DDP-reduced and clipped.')
            with (folder / (stage + '.json')).open('w', encoding='utf-8') as stream:
                json.dump(_summary(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
            #* 首次异常时写出异常前若干步及当前步的中间激活轨迹。
            # 后续连续 NaN 不重复保存相同规模的历史文件，避免大量小文件。
            if (stage == 'after_forward' and self.activation_tracker is not None
                    and not self.saved_activation_history):
                self.activation_tracker.dump(folder / 'activation_history.json')
                self.saved_activation_history = True
            runner.logger.warning('NaN debug: %s, local_abnormal=%s, saved to %s', stage, bool(problems), folder)
        except Exception as exc:
            # 写盘失败也协同退出，不让其他 rank 继续进入 DDP 导致等待。
            error = repr(exc)
            runner.logger.error('NaN debug save failed on rank %s: %s', rank, error)
        failed = self._sync(runner, error is not None)
        if failed:
            raise RuntimeError('NaN debug save failed on at least one rank; see rank logs.')

    def update_loss_streak(self, runner, local_bad):
        """按 iteration 统计全局总 loss 连续异常次数，并在阈值处协同停止。"""
        globally_bad = bool(self._sync(runner, local_bad))
        if globally_bad:
            self.consecutive_bad_losses += 1
            runner.logger.warning(
                'NaN debug: nonfinite total loss streak %d/%d at iteration %d',
                self.consecutive_bad_losses, self.stop_after, runner.iter + 1)
        else:
            if self.consecutive_bad_losses:
                runner.logger.info(
                    'NaN debug: total loss recovered after %d consecutive bad iterations',
                    self.consecutive_bad_losses)
            self.consecutive_bad_losses = 0
        return globally_bad

    def stop_if_persistent(self):
        if self.consecutive_bad_losses >= self.stop_after:
            raise FloatingPointError(
                'Total loss stayed nonfinite for %d consecutive iterations; '
                'diagnostics saved under %s' % (self.stop_after, self.root))

    def state_problems(self, runner):
        # 参数及 BN buffers 都检查；仅 running stats 损坏也会影响验证。
        result = _bad_tensors(
            runner.model.state_dict().items(), self.max_problem_tensors)
        optimizers = runner.optimizer if isinstance(runner.optimizer, dict) else {'optimizer': runner.optimizer}
        for opt_name, optimizer in optimizers.items():
            items = []
            for index, state in enumerate(optimizer.state.values()):
                items.extend(('%s.state.%d.%s' % (opt_name, index, key), value)
                             for key, value in state.items() if torch.is_tensor(value))
            remaining = max(1, self.max_problem_tensors - len(result))
            result.update(_bad_tensors(items, remaining))
        return result

    def maybe_save_rolling_checkpoint(self, runner):
        """周期保存最近模型状态，为首次异常提供相邻的干净恢复点。"""
        if not self.checkpoint_interval:
            return
        iteration = runner.iter + 1
        if iteration % self.checkpoint_interval:
            return

        checkpoint_dir = self.root / 'rolling_checkpoints'
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            #* 保存期间让所有 rank 停在同一位置，避免 rank 0 写盘时其他 rank
            # 提前进入下一次 DDP collective。只有 rank 0 保存完整权重和优化器。
            dist.barrier()
        rank = dist.get_rank() if distributed else 0
        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            filename = 'iter_%08d.pth' % iteration
            runner.save_checkpoint(
                str(checkpoint_dir), filename_tmpl=filename,
                save_optimizer=True,
                meta={'nan_debug_iteration': iteration},
                create_symlink=False,
            )
            checkpoints = sorted(checkpoint_dir.glob('iter_*.pth'))
            for stale in checkpoints[:-self.checkpoint_keep]:
                os.remove(str(stale))
            runner.logger.info(
                'NaN debug: rolling checkpoint saved at iteration %d (%d retained)',
                iteration, min(len(checkpoints), self.checkpoint_keep))
        if distributed:
            dist.barrier()


class _ActivationTracker:
    """低干扰地保留指定模块最近若干步的输出数值范围。"""

    def __init__(self, runner, module_names, history_size):
        if history_size <= 0:
            raise ValueError('nan_debug.activation_history_size must be positive')
        self.history = deque(maxlen=history_size)
        self.pending = []
        self.iteration = None
        root = runner.model.module if hasattr(runner.model, 'module') else runner.model
        modules = dict(root.named_modules())
        missing = [name for name in module_names if name not in modules]
        if missing:
            raise KeyError('NaN debug activation modules not found: %s' % missing)
        self.handles = [modules[name].register_forward_hook(self._hook(name))
                        for name in module_names]

    @staticmethod
    def _tensors(value, path='output'):
        if torch.is_tensor(value):
            yield path, value
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from _ActivationTracker._tensors(item, '%s.%d' % (path, index))
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from _ActivationTracker._tensors(item, '%s.%s' % (path, key))

    def _hook(self, module_name):
        def collect(_module, _inputs, output):
            #* 激活历史只服务于训练NaN定位。评估阶段没有before_train_iter/
            # after_train_iter配对，若继续收集会让pending随验证集长度增长。
            if self.iteration is None:
                return
            for output_name, tensor in self._tensors(output):
                if not (tensor.is_floating_point() or tensor.is_complex()):
                    continue
                value = tensor.detach()
                finite = torch.isfinite(value)
                #* 统计先保留在 GPU；当前前向结束后统一搬到 CPU，只产生一次同步，
                # 避免每个模块分别调用 .item() 导致多次 GPU 流水线停顿。
                finite_value = value.real if value.is_complex() else value
                minimum = torch.where(
                    finite, finite_value, torch.full_like(finite_value, float('inf'))).amin()
                maximum = torch.where(
                    finite, finite_value, torch.full_like(finite_value, float('-inf'))).amax()
                abs_max = torch.where(
                    finite, finite_value.abs(), torch.zeros_like(finite_value)).amax()
                stats = torch.stack([
                    minimum.float(), maximum.float(), abs_max.float(),
                    torch.isnan(value).sum().float(), torch.isinf(value).sum().float(),
                    torch.tensor(value.numel(), device=value.device, dtype=torch.float32),
                ])
                self.pending.append((module_name, output_name, list(value.shape),
                                     str(value.dtype), stats))
        return collect

    def start(self, iteration):
        self.iteration = iteration
        self.pending = []

    def finish(self):
        if not self.pending:
            self.iteration = None
            return
        values = torch.stack([item[-1] for item in self.pending]).cpu().tolist()
        modules = []
        for item, stats in zip(self.pending, values):
            module_name, output_name, shape, dtype, _ = item
            minimum, maximum, abs_max, nan_count, inf_count, numel = stats
            modules.append(dict(
                module=module_name, output=output_name, shape=shape, dtype=dtype,
                min=minimum, max=maximum, abs_max=abs_max,
                nan=int(nan_count), inf=int(inf_count), numel=int(numel),
            ))
        self.history.append(dict(iteration=self.iteration, modules=modules))
        self.pending = []
        # 关闭当前收集窗口；下一次训练迭代由start()重新开启。
        self.iteration = None

    def dump(self, path):
        with Path(path).open('w', encoding='utf-8') as stream:
            json.dump(_summary(list(self.history)), stream, ensure_ascii=False,
                      indent=2, allow_nan=False)


class _BeforeOptimizer(Hook):
    def __init__(self, recorder):
        self.recorder = recorder

    def before_train_iter(self, runner):
        r = self.recorder
        r.scale_before = r.scale()
        if r.activation_tracker is not None:
            r.activation_tracker.start(runner.iter + 1)
        #* 正常阶段不扫描整个模型，避免每步反复读取参数和AdamW状态。
        # 已出现非有限 loss 后，才检查下一步前向之前是否仍有持久状态污染。
        if r.consecutive_bad_losses:
            r.record(runner, 'before_forward', r.state_problems(runner))

    def after_train_iter(self, runner):
        # OptimizerHook 尚未执行，所以这里读取的是本步前向总 loss。
        r = self.recorder
        if r.activation_tracker is not None:
            r.activation_tracker.finish()
        problems = _bad_tensors([('local_total_loss', runner.outputs['loss'])])
        #* 停止条件只统计总 loss，偶发且可恢复的梯度溢出不计入连续 NaN。
        globally_bad = r.update_loss_streak(runner, bool(problems))
        # 复用同一次跨卡检查结果，正常阶段每步只增加一次极小的标志归约。
        r.record(runner, 'after_forward', problems, globally_bad)
        # loss首次异常后立即检查参数、buffer和优化器状态。参数在前向中不会
        # 更新；buffer可能在当前前向中更新，因此仍可区分大部分污染来源。
        if r.consecutive_bad_losses:
            r.record(runner, 'after_forward_state', r.state_problems(runner))
        r.stop_if_persistent()


class _AfterOptimizer(Hook):
    def __init__(self, recorder):
        self.recorder = recorder

    def before_run(self, runner):
        #* Fp16OptimizerHook.before_run()会先执行wrap_fp16_model，把普通卷积
        # 参数转为FP16。本Hook排在其后，将明确标记的FPN_LSS参数及buffer
        # 恢复为FP32，确保局部关闭autocast后输入和卷积权重dtype一致。
        root = runner.model.module if hasattr(runner.model, 'module') else runner.model
        fp32_modules = [module for module in root.modules()
                        if getattr(module, '_nan_debug_force_fp32', False)]
        for module in fp32_modules:
            module.float()
        if fp32_modules:
            runner.logger.warning(
                'NaN debug: restored %d marked module(s) to FP32 after model wrapping.',
                len(fp32_modules))

    def after_train_iter(self, runner):
        r = self.recorder
        # 正常训练期间不做全模型扫描；只在连续异常观察窗口内执行重检查。
        if not r.consecutive_bad_losses:
            r.maybe_save_rolling_checkpoint(runner)
            return
        problems = r.state_problems(runner)
        remaining = max(1, r.max_problem_tensors - len(problems))
        problems.update(_bad_tensors(
            (('gradient.' + name, param.grad)
             for name, param in runner.model.named_parameters()
             if param.grad is not None),
            remaining))
        # 日志缓冲中的最后一个 grad_norm 是本步值，不取 50 步平均后的日志。
        history = runner.log_buffer.val_history.get('grad_norm', [])
        if getattr(r.optimizer_hook, 'grad_clip', None) and history:
            norm = float(history[-1])
            if not math.isfinite(norm):
                problems['grad_norm'] = str(norm)
        scale = r.scale()
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            problems['loss_scale'] = str(scale)
        r.record(runner, 'after_optimizer', problems)


def install_persistent_nan_debug(runner, config):
    """只安装观察 Hook；不替换 step/backward，不修改优化器或模型参数。

    按已有 OptimizerHook 的优先级在其前后各插入一个 Hook，后检查通常
    位于 EMA 之前。当前仅支持逐步更新的标准 OptimizerHook/Fp16OptimizerHook。
    梯度累积会在部分迭代清空梯度，不能用此方案可靠观察，故明确拒绝。
    """
    if not config.get('enabled', False):
        return
    if getattr(runner, '_nan_debug_installed', False):
        raise RuntimeError('NaN debug is already installed.')
    hooks = [h for h in runner.hooks if isinstance(h, OptimizerHook)]
    if len(hooks) != 1 or 'Cumulative' in type(hooks[0]).__name__:
        raise ValueError('NaN debug requires exactly one non-cumulative OptimizerHook.')
    optimizer_hook = hooks[0]
    priority = optimizer_hook.priority
    if not 0 < priority < 100:
        raise ValueError('OptimizerHook priority must allow diagnostics before and after it.')
    recorder = _Recorder(runner, config, optimizer_hook)
    activation_modules = config.get('activation_modules', [])
    if activation_modules:
        #* 只对配置中列出的关键模块注册 hook；环形缓冲仅保存统计量，
        # 不保留激活张量和计算图，因此显存不会随训练步数增长。
        recorder.activation_tracker = _ActivationTracker(
            runner, activation_modules,
            int(config.get('activation_history_size', 50)))
    runner.register_hook(_BeforeOptimizer(recorder), priority=priority - 1)
    runner.register_hook(_AfterOptimizer(recorder), priority=priority + 1)
    runner._nan_debug_installed = True
    runner.logger.info(
        'Persistent NaN debug enabled: %s; save_batch=%s; stop_after=%d; '
        'activation_modules=%d; rolling_checkpoint_interval=%d',
        recorder.root, config.get('save_batch', True), recorder.stop_after,
        len(activation_modules), recorder.checkpoint_interval)
