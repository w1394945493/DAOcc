"""StreamOcc-style infinite batches: each global slot consumes ordered groups."""
import itertools
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Sampler


class GroupInBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=0, world_size=1, seed=0):
        self.dataset = dataset
        self.groups = dataset.sequence_indices
        self.batch_size = int(batch_size)
        self.rank, self.world_size = rank, world_size
        self.seed = int(seed)
        self.global_batch_size = self.batch_size * world_size
        if self.batch_size < 1 or len(self.groups) < self.global_batch_size:
            raise ValueError('Need at least one sequence per global batch slot')
        if not 0 <= rank < world_size or any(not group for group in self.groups):
            raise ValueError('Invalid rank or empty sequence')
        self.size = len(dataset)

    def _groups(self):
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.randperm(len(self.groups), generator=generator).tolist()

    def __iter__(self):
        slots = [self.rank * self.batch_size + i for i in range(self.batch_size)]
        streams = [itertools.islice(self._groups(), slot, None, self.global_batch_size)
                   for slot in slots]
        current = [None] * self.batch_size
        positions = [0] * self.batch_size
        instances = [-1] * self.batch_size
        #*===============================================================================#
        #* (11.12) 沿用 StreamOcc 的每槽位 aug_config；独立 RNG 保证重建迭代器可复现。
        # RNG 位于 sampler 进程，与 DataLoader worker 数量及预取调度无关。
        # 创建的是随机数生成器：每个槽位拥有自己的生成器。它相当于一台“抽签机”，后续调用才会抽出具体参数。
        augmentation_rngs = [np.random.RandomState((self.seed + slot) % (2 ** 32)) for slot in slots] #* 每个batch槽位维护自己的augmentations[i]
        augmentations = [None] * self.batch_size
        while True:
            batch = []
            for i, slot in enumerate(slots):
                if current[i] is None or positions[i] == len(current[i]):
                    current[i] = self.groups[next(streams[i])]
                    positions[i] = 0
                    instances[i] += 1
                    if self.dataset.keep_consistent_seq_aug: # keep_consistent_seq_aug 控制同一子序列内，各帧是否复用同一组几何增强参数：True（当前默认）：子序列开始时随机生成一次，后续帧复用；切换子序列时重新生成。
                        # get_augmentation() 使用生成器，抽取翻转、旋转、缩放等值，返回具体的增强参数字典，存到 augmentations[i]。
                        augmentations[i] = self.dataset.get_augmentation(
                            index=current[i][0], rng=augmentation_rngs[i]) 
                position = positions[i]
                if not self.dataset.keep_consistent_seq_aug:
                    augmentations[i] = self.dataset.get_augmentation(
                        index=current[i][position], rng=augmentation_rngs[i])
                # 构造每帧请求时：复用已保存的增强参数
                batch.append(dict(idx=current[i][position], slot=slot,
                                  sequence_start=position == 0,
                                  sequence_instance=instances[i],
                                  aug_config=deepcopy(augmentations[i])))
                positions[i] += 1
            yield batch

    def __len__(self):
        # Reporting only: iteration is infinite and stopped by IterBasedRunner.
        return (self.size + self.global_batch_size - 1) // self.global_batch_size
