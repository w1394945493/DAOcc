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
        #* (11.12) 沿用 StreamOcc 的每槽位 aug_config；独立 RNG 保证重建迭代器可复现。
        # RNG 位于 sampler 进程，与 DataLoader worker 数量及预取调度无关。
        augmentation_rngs = [np.random.RandomState((self.seed + slot) % (2 ** 32)) for slot in slots]
        augmentations = [None] * self.batch_size
        while True:
            batch = []
            for i, slot in enumerate(slots):
                if current[i] is None or positions[i] == len(current[i]):
                    current[i] = self.groups[next(streams[i])]
                    positions[i] = 0
                    instances[i] += 1
                    if self.dataset.keep_consistent_seq_aug:
                        augmentations[i] = self.dataset.get_augmentation(
                            index=current[i][0], rng=augmentation_rngs[i])
                position = positions[i]
                if not self.dataset.keep_consistent_seq_aug:
                    augmentations[i] = self.dataset.get_augmentation(
                        index=current[i][position], rng=augmentation_rngs[i])
                batch.append(dict(idx=current[i][position], slot=slot,
                                  sequence_start=position == 0,
                                  sequence_instance=instances[i],
                                  aug_config=deepcopy(augmentations[i])))
                positions[i] += 1
            yield batch

    def __len__(self):
        # Reporting only: iteration is infinite and stopped by IterBasedRunner.
        return (self.size + self.global_batch_size - 1) // self.global_batch_size
