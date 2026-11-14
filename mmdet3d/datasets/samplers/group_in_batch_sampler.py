"""StreamOcc-style infinite batches: each global slot consumes ordered groups."""
import itertools
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Sampler


class GroupInBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=0, world_size=1, seed=0, diagnostic_callback=None):
        #* (11.13) 可选只读采样事件，供离线可视化区分随机跳帧与空 GT 过滤。
        self.diagnostic_callback = diagnostic_callback
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
        #*===============================================================#
        #* (11.13) 默认关闭，不消耗原增强 RNG；验证不使用本 sampler。
        self.skip_prob = getattr(dataset, 'skip_prob', 0.0)
        self.sequence_flip_prob = getattr(dataset, 'sequence_flip_prob', 0.0)
        #* (11.13) 参照原 CBGS 初始化一次有放回索引；关闭时保留原采样路径。
        self.cbgs_group_indices = None
        self._cbgs_rng_state = None
        if getattr(dataset, 'sequence_cbgs', False):
            presence = dataset.sequence_class_presence
            if presence is None:
                raise ValueError('Sequence CBGS requires precomputed training class presence')
            pools = [np.flatnonzero(presence[:, c]) for c in range(len(dataset.CLASSES))]
            missing = [name for name, pool in zip(dataset.CLASSES, pools) if not len(pool)]
            if missing:
                raise ValueError('Sequence CBGS cannot sample absent classes: %s' % missing)
            relations = sum(len(pool) for pool in pools)
            rng = np.random.RandomState(self.seed)
            draws = []
            for pool in pools:
                ratio = (1. / len(pools)) / (len(pool) / relations)
                draws.append(rng.choice(pool, int(len(pool) * ratio), replace=True))
            self.cbgs_group_indices = np.concatenate(draws)
            self.cbgs_group_indices.setflags(write=False)
            self._cbgs_rng_state = rng.get_state()
            lengths = np.asarray([
                int(dataset.has_gt_flags[group].sum()) if dataset.skip_empty_gt else len(group)
                for group in self.groups])
            self.size = int(lengths[self.cbgs_group_indices].sum())
            from mmdet3d.utils import get_root_logger
            get_root_logger().info(
                'Sequence CBGS: %d -> %d group entries, %d unique groups, '
                '%d emitted frames per sampled list; runner.max_iters unchanged.',
                len(self.groups), len(self.cbgs_group_indices),
                len(np.unique(self.cbgs_group_indices)), self.size)

    def _groups(self):
        #* (11.13) 所有 rank/slot 复现相同全局组流，再由 islice 分配条目。
        # 只打乱段顺序；重复抽中同一段也走 __iter__ 的新实例/首帧路径。
        if self.cbgs_group_indices is not None:
            rng = np.random.RandomState()
            rng.set_state(self._cbgs_rng_state)
            while True:
                indices = self.cbgs_group_indices.copy()
                rng.shuffle(indices)
                yield from indices.tolist()
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
        emitted = [False] * self.batch_size
        directions = [1] * self.batch_size
        temporal_rngs = [np.random.RandomState((self.seed + slot + 104729) % (2 ** 32))
                         for slot in slots]
        #*===============================================================================#
        #* (11.12) 沿用 StreamOcc 的每槽位 aug_config；独立 RNG 保证重建迭代器可复现。
        # RNG 位于 sampler 进程，与 DataLoader worker 数量及预取调度无关。
        # 创建的是随机数生成器：每个槽位拥有自己的生成器。它相当于一台“抽签机”，后续调用才会抽出具体参数。
        augmentation_rngs = [np.random.RandomState((self.seed + slot) % (2 ** 32)) for slot in slots] #* 每个batch槽位维护自己的augmentations[i]
        augmentations = [None] * self.batch_size
        while True:
            batch = []
            for i, slot in enumerate(slots):
                #* (11.13) 与 StreamOcc 一样每次输出至多随机额外跳一帧。
                # 仅已输出过首帧且至少剩两帧时跳过；空 GT 过滤仍按读取方向继续。
                if (self.skip_prob > 0 and emitted[i] and current[i] is not None
                        and len(current[i]) - positions[i] > 1
                        and temporal_rngs[i].rand() < self.skip_prob):
                    if self.diagnostic_callback is not None:
                        self.diagnostic_callback(dict(reason='random_skip', idx=current[i][positions[i]],
                                                      slot=slot, sequence_instance=instances[i]))
                    positions[i] += 1
                #* (11.12) 空 GT 只推进当前槽位，不改变其他槽位和 worker 内索引。
                # 首个实际输出帧标为 sequence_start；整段为空则继续下一段。
                attempts = 0
                while True:
                    if current[i] is None or positions[i] == len(current[i]):
                        current[i] = self.groups[next(streams[i])]
                        directions[i] = (-1 if self.sequence_flip_prob > 0
                                         and temporal_rngs[i].rand() < self.sequence_flip_prob else 1)
                        if directions[i] == -1:
                            current[i] = current[i][::-1]
                        positions[i] = 0
                        instances[i] += 1
                        emitted[i] = False
                        if self.dataset.keep_consistent_seq_aug:
                            augmentations[i] = self.dataset.get_augmentation(
                                index=current[i][0], rng=augmentation_rngs[i])
                    position = positions[i]
                    if not self.dataset.keep_consistent_seq_aug:
                        augmentations[i] = self.dataset.get_augmentation(
                            index=current[i][position], rng=augmentation_rngs[i])
                    index = current[i][position]
                    positions[i] += 1
                    #* =========================================================#
                    #* (11.12) 原逐帧检查保留；现在只查询初始化时计算的布尔标记。
                    # if (self.dataset.skip_empty_gt and not self.dataset.has_training_gt(index, augmentations[i])):
                    if self.dataset.skip_empty_gt and not self.dataset.has_gt_flags[index]:
                        if self.diagnostic_callback is not None:
                            self.diagnostic_callback(dict(reason='empty_gt', idx=index,
                                                          slot=slot, sequence_instance=instances[i]))
                        attempts += 1 # 这一帧没有GT
                        if attempts >= self.size:
                            raise RuntimeError('No nonempty GT found after a dataset-sized sequential search')
                        continue # 空：顺次检查下一帧 跳到宝珠最近的一层循环 continue：跳过本轮剩余代码，继续最近一层循环的下一轮。
                    
                    # 非空：由GT 发给 worker → 加载图片、点云和 OCC → 执行完整 pipeline
                    batch.append(dict(idx=index, slot=slot,
                                      sequence_start=not emitted[i],
                                      sequence_instance=instances[i],
                                      sequence_direction=directions[i],
                                      aug_config=deepcopy(augmentations[i])))
                    
                    emitted[i] = True
                    break # 推出内层while，再处理下一个槽位
            yield batch

    def __len__(self):
        # Reporting only: iteration is infinite and stopped by IterBasedRunner.
        return (self.size + self.global_batch_size - 1) // self.global_batch_size
