"""本地 CUDA 聚合适配；参考实现只用于小网格检查，不用于实际数据推理。"""
from contextlib import nullcontext
import torch
from torch import nn
from .geometry import grid_centers


class GaussianAggregation(nn.Module):
    def __init__(self, bounds, shape, scale_multiplier=3., backend='cuda'):
        super().__init__()
        if backend not in ('cuda', 'torch_reference'):
            raise ValueError('backend 只能是 cuda 或 torch_reference')
        self.backend, self.shape = backend, tuple(shape)
        self.scale_multiplier = float(scale_multiplier)
        if self.scale_multiplier <= 0:
            raise ValueError('scale_multiplier 必须大于0')
        steps = [(bounds[i+3]-bounds[i])/shape[i] for i in range(3)]
        if max(steps)-min(steps) > 1e-6:
            raise ValueError('原版 localagg_prob 仅支持 XYZ 相同的体素尺寸')
        self.grid_size = steps[0]
        self.register_buffer('queries', grid_centers(bounds, shape).reshape(1, -1, 3), persistent=False)
        self.register_buffer('lower', torch.tensor(bounds[:3]).float(), persistent=False)
        self.cuda_aggregator = None
        self.cuda_args = dict(scale_multiplier=scale_multiplier, H=shape[0], W=shape[1],
                              D=shape[2], pc_min=bounds[:3], grid_size=self.grid_size)

    def _reference(self, means, scales, opacity, semantics, inverse_covariance):
        #* 此实现复核原CUDA公式，包括离散邻域；限制规模，避免误用于整幅Occ3D。
        if means.shape[1]*self.queries.shape[1] > 2_000_000:
            raise ValueError('torch_reference 仅用于小网格检查；实际训练请选择 cuda')
        pts = self.queries[0]
        delta = means[0].unsqueeze(0)-pts.unsqueeze(1)
        power = torch.exp(-.5*torch.einsum('ngi,gij,ngj->ng', delta, inverse_covariance[0], delta))
        qi = ((pts-self.lower)/self.grid_size).int()
        mi = ((means[0].detach()-self.lower)/self.grid_size).int()
        radii = torch.ceil(scales[0].detach().amax(-1)*self.scale_multiplier/self.grid_size).clamp(min=1)
        active = ((qi[:, None]-mi[None]).abs() <= radii[None, :, None]).all(-1)
        power = power*active
        weights = (2*torch.pi)**(-1.5)*torch.det(inverse_covariance[0]).sqrt()*opacity[0]
        weights = power*weights.unsqueeze(0)
        total = weights.sum(-1, keepdim=True)
        semantic = weights @ semantics[0] / total.clamp(min=1e-9)
        fallback = torch.ones_like(semantic)/17
        fallback[..., -1] = 0
        semantic = torch.where(total > 1e-9, semantic, fallback)
        #* 忠实保留上游定义：占据概率不乘opacity，opacity只影响语义混合权重。
        occupied = 1-(1-power).prod(-1)
        return semantic, occupied, power.sum(-1)

    def forward(self, gaussians):
        # 自定义内核仅支持float32；即使外层启用AMP，也不能把半精度指针交给内核。
        context = torch.cuda.amp.autocast(enabled=False) if gaussians['means'].is_cuda else nullcontext()
        with context:
            values = {key: value.float().contiguous() for key, value in gaussians.items()}
            if self.backend == 'cuda':
                if not values['means'].is_cuda:
                    raise RuntimeError('localagg_prob 需要 CUDA；CPU检查请用 torch_reference')
                if self.cuda_aggregator is None:
                    try:
                        from .ops.localagg_prob.local_aggregate_prob import LocalAggregator
                    except ImportError as error:
                        raise RuntimeError('请先在 GaussDAOcc/ops/localagg_prob 中执行 '
                                           'python setup.py build_ext --inplace') from error
                    self.cuda_aggregator = LocalAggregator(**self.cuda_args).to(values['means'].device)
            outputs = []
            #* 上游仅支持batch=1；逐样本调用支持训练批次>1，不丢弃任何样本。
            for index in range(values['means'].shape[0]):
                g = {key: value[index:index+1] for key, value in values.items()}
                if self.backend == 'cuda':
                    semantic, occupied, _ = self.cuda_aggregator(
                        self.queries.float(), g['means'], g['opacity'], g['semantics'],
                        g['scales'], g['inverse_covariance'])
                else:
                    semantic, occupied, _ = self._reference(**g)
                #* 转为18类联合概率，再取log，兼容原DAOcc的CE与get_occ接口。
                occupied = occupied.clamp(0, 1).unsqueeze(-1)
                probability = torch.cat((semantic[:, :17]*occupied, 1-occupied), -1)
                probability = probability.clamp(min=1e-7)
                probability = probability/probability.sum(-1, keepdim=True)
                outputs.append(probability.log().reshape(*self.shape, 18))
            return torch.stack(outputs)
