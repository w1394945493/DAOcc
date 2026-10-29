import numpy as np

from .builder import DATASETS


@DATASETS.register_module()
class CBGSDataset:
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        # dataset 是已经构建好的 NuScenesDatasetOccupancy；CBGS 是外层索引包装器。
        self.dataset = dataset
        # resample=True 时 __getitem__/__len__ 使用 CBGS 索引；False 时直接透传原数据集。
        self.resample = dataset.resample
        # 这里的 CLASSES 是 nuScenes 的 10 个目标检测类别，不是 occupancy 的体素类别。
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        # 包装器初始化时生成一次重采样索引。当前 set_epoch() 不会重新生成该索引，
        # 但训练 DataLoader 的 sampler 仍可在每个 epoch 打乱这些外层索引的读取顺序。
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, "flag") and self.resample:
            # group flag 也必须按重采样后的索引展开，供分组采样器正确组织 batch。
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices], dtype=np.uint8
            )
        else:
            self.flag = self.dataset.flag
    
    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def _get_sample_indices(self):
        """按照检测类别对场景进行 CBGS 重采样，返回重采样后的原场景索引。

        Returns:
            list[int]: 重采样后的原始数据集索引。索引可以重复；重复表示同一个
            原始场景会在一个 epoch 中被训练多次。
        """
        # 整体流程：
        # 1. 遍历原始数据集，得到每个检测类别对应的场景索引列表。
        #    例如 car -> [0, 1, 2]，pedestrian -> [2, 3]；同一场景可以属于多类。
        # 2. 用各类别列表长度计算当前类别占比 p_i。
        # 3. 将每个类别的目标占比设为 1 / 类别数 C，计算采样倍率：
        #       ratio_i = (1 / C) / p_i
        # 4. 从类别 i 的场景列表中有放回地抽取
        #       int(该类别原场景数 * ratio_i)
        #    个索引。该数量约等于 duplicated_samples / C，所以每类贡献的
        #    索引数大致相同：常见类会少抽，稀有类的场景可能被重复抽取。
        # 5. 将所有类别抽到的索引依次拼接成 sample_indices 并返回。
        #    后续 __getitem__(外层索引) 会先查询 sample_indices[外层索引]，
        #    再用得到的原场景索引访问 self.dataset；DataLoader 会负责打乱读取顺序。
        #
        # 这里平衡的是“包含某个 3D 检测类别的场景采样机会”，不是框实例数量，
        # 也不是 occupancy 的体素语义类别。由于多类别场景会进入多个类别列表，
        # 同一场景可能从不同列表被选中，最终类别分布是近似均衡而非严格均衡。

        # 为每个检测类别建立“包含该类别的原始场景索引”列表。
        # 同一场景若同时含 car 和 pedestrian，会分别进入两个类别列表；
        # 因而 duplicated_samples 是按“场景-类别出现关系”计数，不是原始场景数。
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            # get_cat_ids() 只返回该场景中实际出现过的检测类别，每类最多记一次。
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        # 计算重采样前每个类别在全部“场景-类别关系”中的占比 p_i。
        class_distribution = {k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()}

        sample_indices = []

        # 目标占比设为 1 / 类别数；类别 i 的采样倍率为 target / p_i。
        # 稀有类别倍率通常大于 1，np.random.choice 默认有放回采样，会重复稀有场景；
        # 常见类别倍率通常小于 1，会减少其进入当前重采样列表的次数。
        # 多类别场景可能通过不同类别列表被重复选中，所以最终是近似类别均衡。
        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices #重采样后的原始数据集索引。索引可以重复；重复表示同一原始场景会在一个 epoch 中被训练多次。

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.resample:
            # DataLoader 给出的是 CBGS 外层索引；先映射回原始数据集索引再读取样本。
            ori_idx = self.sample_indices[idx]
            return self.dataset[ori_idx]
        else:
            return self.dataset[idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        if self.resample:
            # 一个 epoch 的样本数由重采样索引长度决定，可能不同于原数据集长度。
            return len(self.sample_indices)
        else:
            return len(self.dataset)
