"""One detached BEV batch; ordinary runtime state, excluded from state_dict."""
from copy import deepcopy
import torch


class BEVFeatureCache:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.bev = None
        self.metas = None
        self.valid = None

    def get(self, metas):
        if not self.enabled or self.bev is None:
            return None, None
        if len(metas) != self.bev.shape[0]:
            self.reset()
            return None, None
        valid = [not cur['sequence_start']
                 and cur['slot'] == old['slot']
                 and cur['sequence_instance'] == old['sequence_instance']
                 and cur['scene_token'] == old['scene_token']
                 #* (11.13) 反序仅允许显式 direction=-1 的同实例历史；默认仍严格正序。
                 and cur.get('sequence_direction', 1) in (-1, 1)
                 and cur.get('sequence_direction', 1) == old.get('sequence_direction', 1)
                 and (cur['timestamp'] - old['timestamp']) * cur.get('sequence_direction', 1) > 0
                 for cur, old in zip(metas, self.metas)]
        self.valid = torch.tensor(valid, dtype=torch.bool, device=self.bev.device)
        # Invalid slots are explicitly zeroed and must also be masked by future fusion.
        self.bev[~self.valid] = 0
        return self.bev, self.valid

    def update(self, bev, metas):
        if not self.enabled:
            return
        if bev.ndim != 4 or bev.shape[0] != len(metas):
            raise ValueError('Expected BEV [B,C,X,Y] matching stream_metas')
        self.bev = bev.detach().clone()
        #* (11.12) 位姿及增强矩阵是嵌套 list，缓存持有独立快照。
        self.metas = deepcopy(metas)
        self.valid = torch.ones(bev.shape[0], dtype=torch.bool, device=bev.device)
