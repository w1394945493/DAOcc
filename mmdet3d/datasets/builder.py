import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")


def build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import ClassBalancedDataset, ConcatDataset, RepeatDataset

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        # RepeatDataset: 将内层数据集重复读取若干遍；不在磁盘上复制数据，仅重复索引
        # 常用于：（1）数据集较少，希望每个epoch增加迭代次数。（2）多数据集联合训练时，提高某个数据集的采样频率
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset":
        # MMDectection通用类别均衡包装器：根据类别出现频率，为包含稀有类别的图片或场景设置更高的重复次数
        # "oversample_thr"：稀有类别判断阈值。类别出现频率越低，包含该类别的样本越可能被重复。
        # 与CBGS计算方法不同：ClassBalancedDataset：根据类别频率和 oversample_thr 计算样本重复因子（保留所有原样本，额外重复含稀有类别的样本。）
        # CBGSDataset：把目标类别采样比例尽量调整到 1 / 类别数（分别按类别重新抽样，让每个类别贡献大致相同数量的场景；常见类别可能被少采，稀有类别会被重复采。）
        # | 对比 | `ClassBalancedDataset` | `CBGSDataset` |
        # |---|---|---|
        # | 常见类别样本 | 通常全部保留 | 可能减少采样 |
        # | 稀有类别样本 | 根据阈值适量重复 | 重复到接近各类别等量 |
        # | 控制参数 | `oversample_thr` | 当前实现没有额外阈值 |
        # | 均衡强度 | 相对温和 | 通常更强 |
        # | 目标 | 缓解稀有类别采样不足 | 让各类别采样贡献接近一致 |
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    elif cfg["type"] == "CBGSDataset":
        # 配置形态为 data.train = {type: CBGSDataset, dataset: {...}}。
        # 先递归构建内层 NuScenesDatasetOccupancy，再用 CBGSDataset 包装它。
        # 包装器只改变 DataLoader 看到的长度及“外层索引 -> 原样本索引”映射，
        # 不复制标注文件，也不修改内层数据预处理 pipeline。
        dataset = CBGSDataset(build_dataset(cfg["dataset"], default_args))
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
