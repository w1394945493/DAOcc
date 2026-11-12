"""StreamOcc validation sequence partitioning, one stream per rank, no padding."""
from torch.utils.data import Sampler


class SceneDistributedSampler(Sampler):
    def __init__(self, dataset, rank=0, world_size=1):
        if not 0 <= rank < world_size:
            raise ValueError('Invalid rank/world size')
        #* (11.12) 沿用 StreamOcc 验证采样器的边界判定：时间差 >4 秒或 LiDAR 文件名前缀变化。
        # 不使用训练子序列 flag，不打乱，不补重复帧，每卡 batch size 固定为 1。
        groups = []
        previous_time, previous_vehicle = None, None
        for index, info in enumerate(dataset.data_infos):
            timestamp = info['timestamp'] / 1e6
            vehicle = info['lidar_path'].split('/')[-1][:4] if 'lidar_path' in info else None
            if index == 0 or abs(timestamp - previous_time) > 4 or vehicle != previous_vehicle:
                groups.append([])
            groups[-1].append(index)
            previous_time, previous_vehicle = timestamp, vehicle
        split_length = len(dataset) // world_size
        prefix = 0
        self.requests = []
        for group_id, indices in enumerate(groups):
            # 最后一个 rank 接收尾部余数；避免 StreamOcc floor 划分在极端短序列时漏掉尾帧。
            owner = min(prefix // split_length, world_size - 1) if split_length else world_size - 1
            if owner == rank:
                for position, index in enumerate(indices):
                    self.requests.append(dict(idx=index, slot=rank, sequence_start=position == 0,
                                              sequence_instance=group_id))
            prefix += len(indices)

    def __iter__(self):
        return iter(self.requests)

    def __len__(self):
        return len(self.requests)
