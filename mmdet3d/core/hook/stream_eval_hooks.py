"""Training-time streaming validation: BS=1, independent history, original metrics."""
import os
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import scatter
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core import EvalHook, DistEvalHook


def collect_stream_results(local_results, size, tmpdir=None):
    """Collect indexed results without the default interleaving assumption."""
    distributed = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if world > 1:
        directory = [None]
        if rank == 0:
            if tmpdir is not None:
                os.makedirs(tmpdir, exist_ok=True)
            directory[0] = tempfile.mkdtemp(prefix='stream_eval_', dir=tmpdir)
        dist.broadcast_object_list(directory, src=0)
        mmcv.dump(local_results, os.path.join(directory[0], f'part_{rank}.pkl'))
        dist.barrier()
        if rank != 0:
            return None
        try:
            parts = [mmcv.load(os.path.join(directory[0], f'part_{i}.pkl')) for i in range(world)]
        finally:
            shutil.rmtree(directory[0])
    else:
        parts = [local_results]
    ordered = [None] * size
    for part in parts:
        for index, prediction in part:
            if not 0 <= index < size or ordered[index] is not None:
                raise RuntimeError(f'Duplicate or invalid validation index: {index}')
            ordered[index] = prediction
    if any(item is None for item in ordered):
        raise RuntimeError('Streaming validation did not cover every dataset frame')
    return ordered


class StreamEvalMixin:
    def __init__(self, *args, evaluate_last=True, **kwargs):
        kwargs['by_epoch'] = False
        super().__init__(*args, **kwargs)
        self.evaluate_last = evaluate_last

    def _should_evaluate(self, runner):
        return (super()._should_evaluate(runner) or
                (self.evaluate_last and runner.iter + 1 == runner.max_iters))

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        wrapped = runner.model
        model = wrapped.module if hasattr(wrapped, 'module') else wrapped
        modes = [(module, module.training) for module in wrapped.modules()]
        #* (11.12) 沿用 StreamOcc 连续序列验证；DAOcc 保留当前权重评估，单独切换历史缓存。
        # 各 rank 帧数可不同，直接调用本地模型，避免 DDP 前向同步因不等长验证而阻塞。
        if distributed and getattr(self, 'broadcast_bn_buffer', True):
            for module in model.modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_mean, 0)
                    dist.broadcast(module.running_var, 0)
        try:
            wrapped.eval()
            local_results = []
            with model.validation_stream(), torch.no_grad():
                for batch in self.dataloader:
                    data = scatter(batch, [torch.cuda.current_device()])[0]
                    metas = data['stream_metas']
                    if len(metas) != 1:
                        raise ValueError('Streaming validation requires one sample per rank')
                    predictions = model(return_loss=False, rescale=True, **data)
                    if len(predictions) != 1:
                        raise RuntimeError('Expected one validation prediction per frame')
                    local_results.append((metas[0]['dataset_index'], predictions[0]))
            results = collect_stream_results(local_results, len(self.dataloader.dataset),
                                             getattr(self, 'tmpdir', None))
            self.latest_results = results
            if rank == 0:
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                key_score = self.evaluate(runner, results)
                if self.save_best and key_score is not None:
                    self._save_ckpt(runner, key_score)
        finally:
            for module, training in modes:
                module.training = training
        if distributed:
            dist.barrier()


class StreamEvalHook(StreamEvalMixin, EvalHook):
    pass


class StreamDistEvalHook(StreamEvalMixin, DistEvalHook):
    pass
