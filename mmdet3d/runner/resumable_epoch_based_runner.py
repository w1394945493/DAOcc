"""独立迭代保存模式：epoch组织数据，支持全局步数上限和轮内续训。"""
import hashlib
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from mmcv.runner import RUNNERS, get_dist_info
from mmcv.runner.checkpoint import save_checkpoint, weights_to_cpu, get_state_dict

from .epoch_based_runner import CustomEpochBasedRunner


def dataset_state(dataset):
    #* 保存CBGS实际映射；token哈希用于识别“长度相同但样本顺序已变”的数据集。
    state = dict(type=type(dataset).__name__, length=len(dataset))
    if hasattr(dataset, 'data_infos'):
        ids = [str(x.get('token', x.get('sample_idx', i)))
               for i, x in enumerate(dataset.data_infos)]
        state['tokens_sha256'] = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if hasattr(dataset, 'flag'):
        state['flag_sha256'] = hashlib.sha256(np.asarray(dataset.flag).tobytes()).hexdigest()
    if hasattr(dataset, 'sample_indices'):
        state['sample_indices'] = list(dataset.sample_indices)
        state['resample'] = bool(dataset.resample)
    if hasattr(dataset, 'dataset'):
        state['child'] = dataset_state(dataset.dataset)
    if hasattr(dataset, 'datasets'):
        state['children'] = [dataset_state(d) for d in dataset.datasets]
    return state


def restore_dataset(dataset, state):
    if type(dataset).__name__ != state['type']:
        raise ValueError('恢复失败：数据集包装类型改变')
    if 'child' in state:
        restore_dataset(dataset.dataset, state['child'])
    if 'children' in state:
        if len(dataset.datasets) != len(state['children']):
            raise ValueError('恢复失败：子数据集数量改变')
        for d, s in zip(dataset.datasets, state['children']):
            restore_dataset(d, s)
    if 'sample_indices' in state:
        if bool(dataset.resample) != state['resample']:
            raise ValueError('恢复失败：CBGS启用状态改变')
        indices = np.asarray(state['sample_indices'], dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= len(dataset.dataset)):
            raise ValueError('恢复失败：CBGS索引越界')
        dataset.sample_indices = indices.tolist()
        dataset.flag = (np.asarray(dataset.dataset.flag)[indices]
                        if dataset.resample else dataset.dataset.flag)
    actual = dataset_state(dataset)
    for key in ('length', 'tokens_sha256', 'flag_sha256'):
        if actual.get(key) != state.get(key):
            raise ValueError(f'恢复失败：数据集{key}改变')


def prepare_resumable_datasets(datasets, resume_from=None):
    """必须在构建Sampler前恢复CBGS，否则Sampler缓存的长度/分组信息会过期。"""
    states = None
    if resume_from:
        checkpoint = torch.load(resume_from, map_location='cpu')
        resume = checkpoint.get('meta', {}).get('resume_state')
        if resume is None:
            raise ValueError('轮内续训需要新模式checkpoint；旧权重请使用load_from开始新训练')
        states = resume['datasets']
        del checkpoint
    else:
        # 分布式初次运行统一CBGS映射，随后每个rank由Sampler划分不同batch。
        states = [dataset_state(d) for d in datasets]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            objects = [states if get_dist_info()[0] == 0 else None]
            torch.distributed.broadcast_object_list(objects, src=0)
            states = objects[0]
    if len(states) != len(datasets):
        raise ValueError('恢复失败：训练数据集数量改变')
    for dataset, state in zip(datasets, states):
        restore_dataset(dataset, state)


class EpochOffsetBatchSampler:
    """确定每轮索引顺序后直接跳过已完成batch，不加载/增强被跳过的数据。"""
    def __init__(self, batch_sampler, seed):
        self.base = batch_sampler
        self.sampler = batch_sampler.sampler
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0

    def __len__(self):
        # 返回完整轮长度，确保日志、进度和checkpoint兼容性校验保持不变。
        return len(self.base)

    def __iter__(self):
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self.epoch)
        # 单卡GroupSampler使用numpy全局随机数；暂存/还原，避免影响增强随机性。
        np_state, py_state = np.random.get_state(), random.getstate()
        cpu_state = torch.get_rng_state()
        try:
            seed = (self.seed + self.epoch) % (2 ** 32)
            np.random.seed(seed)
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            batches = list(self.base)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.set_rng_state(cpu_state)
        yield from batches[self.start_batch:]


def resumable_loader(loader, seed):
    if seed is None:
        raise ValueError('迭代续训要求配置非空seed以固定样本排列')
    # 保留原来的分组/分布式Sampler和collate，不改样本增强；关闭持久worker以更新epoch。
    return DataLoader(
        loader.dataset, batch_sampler=EpochOffsetBatchSampler(loader.batch_sampler, seed),
        num_workers=loader.num_workers, collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory, timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn, generator=loader.generator,
        persistent_workers=False,
    )


@RUNNERS.register_module()
class ResumableEpochBasedRunner(CustomEpochBasedRunner):
    def __init__(self, *args, max_iters=None, **kwargs):
        # MMCV父类不允许同时给max_epochs/max_iters，因此单独保存全局步数上限。
        super().__init__(*args, **kwargs)
        if max_iters is not None and (not isinstance(max_iters, int) or max_iters <= 0):
            raise ValueError('max_iters必须为正整数或null')
        self.iter_limit = max_iters
        self._resume_offset = 0
        self._resume_state = None
        self._resume_ema = None
        self._completed_in_epoch = 0
        self._in_train_iter = False

    def resume(self, checkpoint, resume_optimizer=True, map_location='cpu'):
        # 独立恢复，不沿用EpochBasedRunner把未完成epoch当作完成轮数的语义。
        state = self.load_checkpoint(checkpoint, map_location=map_location, strict=True)
        meta = state['meta']
        if 'resume_state' not in meta:
            raise ValueError('不支持将旧epoch权重作为轮内断点；请用load_from')
        self._resume_state = meta['resume_state']
        self._epoch, self._iter = meta['epoch'], meta['iter']
        self._resume_offset = self._resume_state['inner_iter_completed']
        self._resume_ema = meta.pop('ema_state', None)
        self.meta = meta
        if resume_optimizer:
            if 'optimizer' not in state:
                raise ValueError('断点缺少optimizer状态，无法完整续训')
            if isinstance(self.optimizer, dict):
                for key, optimizer in self.optimizer.items():
                    optimizer.load_state_dict(state['optimizer'][key])
            else:
                self.optimizer.load_state_dict(state['optimizer'])
        self.logger.info('恢复：epoch索引=%d，轮内已完成=%d，全局已完成=%d',
                         self.epoch, self._resume_offset, self.iter)

    def _loader_signature(self, loader):
        sampler = loader.batch_sampler.sampler
        return dict(world_size=self.world_size, iters_per_epoch=len(loader),
                    batch_size=loader.batch_sampler.base.batch_size,
                    drop_last=loader.batch_sampler.base.drop_last,
                    sampler_type=type(sampler).__name__,
                    sampler_seed=getattr(sampler, 'seed', None),
                    order_seed=loader.batch_sampler.seed)

    def run(self, data_loaders, workflow, **kwargs):
        if len(data_loaders) != 1 or workflow != [('train', 1)]:
            raise ValueError('新Runner仅支持一个训练loader和[(train,1)]工作流')
        self.data_loader = data_loaders[0]
        if not isinstance(self.data_loader.batch_sampler, EpochOffsetBatchSampler):
            raise ValueError('必须使用resumable_loader构建新模式DataLoader')
        if not len(self.data_loader):
            raise ValueError('训练DataLoader为空')
        self._signature = self._loader_signature(self.data_loader)
        if self._resume_state and self._resume_state['loader'] != self._signature:
            raise ValueError(f'恢复条件改变：原{self._resume_state["loader"]}，现{self._signature}')
        if self.iter != self.epoch * len(self.data_loader) + self._resume_offset:
            raise ValueError('checkpoint全局步数与epoch/轮内位置不一致')
        self._max_iters = min(self.max_epochs * len(self.data_loader),
                              self.iter_limit or self.max_epochs * len(self.data_loader))
        if self.iter > self.max_iters:
            raise ValueError('断点已完成步数超过新的训练上限')
        self._dataset_state = [dataset_state(d) for d in self._dataset]
        # 不保存累积中的梯度：显式拒绝累积Hook，防止看似恢复成功但更新不一致。
        if any('Cumulative' in type(h).__name__ for h in self.hooks):
            raise ValueError('当前轮内续训模式不支持梯度累积Hook')
        self.logger.info('迭代模式：最多%d epochs / %d iterations；样本顺序恢复，增强继续随机',
                         self.max_epochs, self.max_iters)
        self.call_hook('before_run')
        while self.epoch < self.max_epochs and self.iter < self.max_iters:
            self.train(self.data_loader, **kwargs)
        self.call_hook('after_run')

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        for dataset in self._dataset:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(self.epoch)
        sampler = data_loader.batch_sampler
        sampler.epoch = self.epoch
        sampler.start_batch = self._resume_offset
        start = self._resume_offset
        self._resume_offset = 0
        self._completed_in_epoch = start
        self.call_hook('before_train_epoch')
        for i, data in enumerate(data_loader, start=start):
            self._inner_iter = i
            self.data_batch = data
            self.call_hook('before_train_iter')
            self.run_iter(data, train_mode=True, **kwargs)
            self._completed_in_epoch = i + 1
            self._in_train_iter = True
            self.call_hook('after_train_iter')
            self._in_train_iter = False
            del self.data_batch
            self._iter += 1
            if self.iter >= self.max_iters:
                break
        if self._completed_in_epoch == len(data_loader):
            self.call_hook('after_train_epoch')
            self._epoch += 1
            self._completed_in_epoch = 0
        # max_iters在轮中到达时，不伪造一个已完成epoch。

    def save_checkpoint(self, out_dir, filename_tmpl='iter_{}.pth', save_optimizer=True,
                        meta=None, create_symlink=True):
        completed = self.iter + int(self._in_train_iter)
        inner = self._completed_in_epoch
        epoch = self.epoch
        if inner == len(self.data_loader):
            epoch, inner = epoch + 1, 0
        metadata = deepcopy(self.meta or {})
        metadata.update(meta or {})
        metadata.update(epoch=epoch, iter=completed)
        metadata['resume_state'] = dict(version=1, inner_iter_completed=inner,
                                       loader=self._signature, datasets=self._dataset_state)
        #* EMA与模型/优化器/GradScaler元数据写入同一文件，避免独立文件错步或丢失。
        if hasattr(self, 'ema_model'):
            metadata['ema_state'] = dict(
                state_dict=weights_to_cpu(get_state_dict(self.ema_model.ema)),
                updates=self.ema_model.updates)
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename_tmpl.format(completed)
        temporary = directory / (path.name + '.tmp')
        save_checkpoint(self.model, str(temporary),
                        optimizer=self.optimizer if save_optimizer else None, meta=metadata)
        os.replace(temporary, path)
        if create_symlink:
            link = directory / 'latest.pth.tmp'
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(path.name)
            os.replace(link, directory / 'latest.pth')
