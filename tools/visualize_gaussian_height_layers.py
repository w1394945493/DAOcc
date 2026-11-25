#!/usr/bin/env python3
"""按中心Z唯一分层绘制高斯XY投影，仅保存PNG（2026-11-24新增）。"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def layer_ids(z, edges):
    """左闭右开，最后一层包含上界；范围外/非有限值编号为-1。"""
    import numpy as np
    ids = np.searchsorted(edges, z, side='right') - 1
    ids[z == edges[-1]] = len(edges) - 2
    ids[(z < edges[0]) | (z > edges[-1]) | ~np.isfinite(z)] = -1
    return ids


def draw_layers(path, means, cov, opacity, edges, bounds, threshold, title):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    active = opacity > threshold
    means, cov, opacity = means[active], cov[active], opacity[active]
    if not all(np.isfinite(x).all() for x in (means, cov, opacity)):
        raise ValueError('有效高斯含非有限中心、协方差或opacity')
    ids = layer_ids(means[:, 2], edges)
    #! 按中心高度唯一归属；绘制的是整颗高斯的XY投影，不是三维高斯截面。
    sigma = np.zeros((len(cov), 2, 2))
    sigma[:, 0, 0] = cov[:, 0] + .3 * bounds[0][2] ** 2
    sigma[:, 0, 1] = sigma[:, 1, 0] = cov[:, 1]
    sigma[:, 1, 1] = cov[:, 3] + .3 * bounds[1][2] ** 2
    eig, vec = np.linalg.eigh(sigma)
    if (eig <= 0).any():
        raise ValueError('高斯投影协方差非正定')
    angle = np.linspace(0, 2 * np.pi, 33)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    contours = means[:, :2, None] + vec @ (3 * np.sqrt(eig)[:, :, None] * circle)
    layers = len(edges) - 1
    cols = min(5, layers)
    fig, axes = plt.subplots((layers + cols - 1) // cols, cols,
                             figsize=(5 * cols, 5 * ((layers + cols - 1) // cols)),
                             squeeze=False)
    norm = Normalize(0, 1)
    for i, ax in enumerate(axes.flat):
        if i >= layers:
            ax.axis('off')
            continue
        selected = ids == i
        collection = LineCollection(contours[selected].transpose(0, 2, 1),
                                    cmap='viridis', norm=norm, linewidths=.5)
        collection.set_array(opacity[selected])
        ax.add_collection(collection)
        closing = ']' if i == layers - 1 else ')'
        ax.set_title(f'Z in [{edges[i]:.2f}, {edges[i+1]:.2f}{closing} m | N={selected.sum()}')
        ax.set_xlim(bounds[0][:2]); ax.set_ylim(bounds[1][:2])
        ax.set_aspect('equal'); ax.grid(alpha=.2)
        ax.set_xlabel('LiDAR X (m)'); ax.set_ylabel('LiDAR Y (m)')
    fig.suptitle(title + f'\nActive={len(ids)} | in range={(ids >= 0).sum()} | '
                 f'outside={(ids < 0).sum()} | opacity > {threshold:g}', fontsize=13)
    fig.tight_layout(rect=[0, .05, .94, .93])
    cax = fig.add_axes([.95, .18, .01, .6])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax, label='Opacity')
    fig.text(.02, .02, 'Unique center-Z assignment; XY 3-sigma envelopes with renderer low-pass. '
             'Not height-sliced features or 3D cross-sections.', fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_gaussian_bev_iter.yaml')
    p.add_argument('--checkpoint', type=Path, help='省略时使用随机初始化')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/gaussian_height_layers')
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--num-samples', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--z-range', nargs=2, type=float, default=[-5., 3.])
    p.add_argument('--num-layers', type=int, default=10)
    a, overrides = p.parse_known_args()
    #! (2026-11-24) 支持训练命令同款点路径覆盖，加载权重前构建对应的几何预测分支。
    if len(overrides) % 2:
        p.error('配置覆盖须使用 --model.xxx value 成对传入')
    model_overrides = []
    for key, value in zip(overrides[::2], overrides[1::2]):
        if not key.startswith('--model.'):
            p.error('未知参数：' + key + '；配置覆盖仅支持 --model.xxx')
        if value.lower() in ('true', 'false'):
            value = str(value.lower() == 'true')
        model_overrides.extend([key, value])
    import numpy as np
    if (a.num_layers < 1 or a.num_samples < 1 or a.start_index < 0
            or not np.isfinite(a.z_range).all() or a.z_range[0] >= a.z_range[1]):
        p.error('高度范围须有限且递增，层数/样本数须为正，起始索引须非负')
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from torchpack.utils.config import Config as TorchConfig
    from mmdet.apis import set_random_seed
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    import mmdet3d.models.vtransforms.gaussian_bev
    from compare_gaussian_bev import (clear_pretrained, uniformly_spaced_indices,
                                      load_checkpoint_strict_except, suppress_external_output)

    torch.cuda.set_device(a.gpu)
    set_random_seed(0)
    raw = TorchConfig(); raw.load(str(a.config.resolve()), recursive=True)
    raw.update(model_overrides)
    cfg = Config(recursive_eval(raw))
    vt_cfg = cfg.model.encoders.camera.vtransform
    if vt_cfg.type != 'GaussianBEVTransform':
        raise ValueError('配置必须使用GaussianBEVTransform')
    data_cfg = deepcopy(cfg.data.val)
    data_cfg.test_mode = True; data_cfg.pop('samples_per_gpu', None)
    model_cfg = deepcopy(cfg.model); clear_pretrained(model_cfg)
    with suppress_external_output():
        dataset = build_dataset(data_cfg)
        model = build_model(model_cfg)
        if a.checkpoint:
            prefix = 'encoders.camera.vtransform.'
            buffers = [prefix + k for k in ('geometry.fixed_scale', 'geometry.scale_min',
                       'geometry.scale_max', 'geometry.offset_limit', 'geometry.covariance',
                       'renderer.view', 'renderer.projection')]
            load_checkpoint_strict_except(model, a.checkpoint, buffers)
        else:
            model.init_weights()
    model.float().eval().cuda(a.gpu)
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    indices = uniformly_spaced_indices(len(dataset), a.start_index, a.num_samples)
    edges = np.linspace(*a.z_range, a.num_layers + 1)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = model.encoders['camera']['vtransform'].renderer
    with torch.no_grad():
        for index in indices:
            data = scatter(collate([dataset[index]], samples_per_gpu=1), [a.gpu])[0]
            captured = []
            def capture(_module, inputs):
                #! 捕获实际渲染输入，因此包含可学习几何产生的中心修正和协方差。
                captured.append(tuple(t[0].detach().float().cpu().numpy() for t in inputs[1:4]))
            hook = renderer.register_forward_pre_hook(capture)
            keys = ('points', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image',
                    'camera_intrinsics', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix')
            try:
                model.extract_camera_features(data['img'], *[data[k] for k in keys],
                    data.get('metas', data.get('img_metas')),
                    **{k: data[k] for k in ('camera_ego2global',) if k in data})
            finally:
                hook.remove()
            if len(captured) != 1:
                raise RuntimeError('预期每帧捕获一次高斯渲染输入')
            means, cov, opacity = captured[0]
            token = dataset.data_infos[index].get('token', str(index))
            path = a.output_dir / f'{index:06d}_{token}.png'
            draw_layers(path, means, cov, opacity[:, 0], edges,
                        (vt_cfg.xbound, vt_cfg.ybound), renderer.threshold,
                        f'Gaussian height layers | {index} | ' +
                        ('checkpoint' if a.checkpoint else 'RANDOM INITIALIZATION'))
            print(f'Saved: {path}', flush=True)


if __name__ == '__main__':
    main()
