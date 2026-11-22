#!/usr/bin/env python3
"""同帧比较 BEVTransformV2 / GaussianBEVTransform 的推理开销及 BEV 特征。

从 DAOCC 根目录运行（两份 checkpoint 均可省略，省略时明确标记随机初始化）：
python tools/compare_gaussian_bev.py --output-dir outputs/bev_compare \
    --baseline-checkpoint BASE.pth --gaussian-checkpoint GAUSSIAN.pth

默认 FP32、batch=1；只执行编码和 ConvFuser，不执行 decoder/任务头。
启用性能统计时，计时含 CPU 调度和 GPU 执行，不含加载、数据搬运、可视化；
使用 --skip-performance 可跳过耗时和显存统计，仅执行一次前向并保存可视化。
每个样本将特征对比、高斯3σ椭圆和精确累计alpha覆盖范围保存为一张图。
"""
import argparse
import csv
import gc
import os
import json
import logging
import random
import re
import sys
import time
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class suppress_external_output:
    """Hide verbose third-party config/model construction output."""

    def __enter__(self):
        self.stream = open(os.devnull, 'w')
        self.stdout = redirect_stdout(self.stream)
        self.stderr = redirect_stderr(self.stream)
        self.logging_disable_level = logging.root.manager.disable
        # MMCV log handlers may already hold the original stderr stream, so
        # stream redirection alone cannot hide init_weights parameter reports.
        logging.disable(logging.CRITICAL)
        self.stdout.__enter__()
        self.stderr.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stderr.__exit__(exc_type, exc_value, traceback)
        self.stdout.__exit__(exc_type, exc_value, traceback)
        logging.disable(self.logging_disable_level)
        self.stream.close()
        # Construction errors still propagate and produce a normal traceback.
        return False


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    directory = ROOT / 'configs/nuscenes/occ3d'
    p.add_argument('--baseline-config', type=Path, default=directory / 'daocc_occ3d_nus_w_mask_inin_spconv2_voxel_iter.yaml')
    p.add_argument('--gaussian-config', type=Path, default=directory / 'daocc_occ3d_nus_w_mask_inin_spconv2_voxel_gaussian_bev_iter.yaml')
    p.add_argument('--baseline-checkpoint', type=Path)
    p.add_argument('--gaussian-checkpoint', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num-samples', type=int, default=10,
                   help='从验证集可用区间内均匀抽取的样本数')
    p.add_argument('--start-index', type=int, default=0,
                   help='均匀抽样区间的起始索引（终点始终为验证集末尾）')
    p.add_argument('--warmup', type=int, default=5, help='每个样本每个测量阶段预热次数')
    p.add_argument('--repeat', type=int, default=10, help='每个样本每阶段计时次数')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--feature-scale', choices=['independent', 'shared'], default='independent')
    performance = p.add_mutually_exclusive_group()
    performance.add_argument('--measure-performance', dest='measure_performance',
                             action='store_true', help='统计各阶段耗时和峰值显存（默认）')
    performance.add_argument('--skip-performance', dest='measure_performance',
                             action='store_false', help='跳过耗时和显存统计，仅前向并可视化')
    p.set_defaults(measure_performance=True)
    p.add_argument('--max-gaussians-vis', type=int, default=0,
                   help='椭圆图最多绘制的高斯数；0表示绘制全部，超限时按opacity保留')
    a = p.parse_args()
    if (a.num_samples < 1 or a.repeat < 1 or a.warmup < 0 or
            a.start_index < 0 or a.max_gaussians_vis < 0):
        p.error('num-samples/repeat必须为正；warmup/start-index/max-gaussians-vis不能为负')
    return a


def clear_pretrained(value):
    if isinstance(value, dict):
        if 'pretrained' in value:
            value['pretrained'] = None
        if isinstance(value.get('init_cfg'), dict) and value['init_cfg'].get('type') == 'Pretrained':
            value['init_cfg'] = None
        for item in value.values():
            clear_pretrained(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            clear_pretrained(item)


def uniformly_spaced_indices(dataset_size, start_index, num_samples):
    """Select unique, approximately equally spaced indices through the remaining set."""
    available = dataset_size - start_index
    if available <= 0:
        raise ValueError(
            f'start-index {start_index} is outside dataset with {dataset_size} samples')
    if num_samples > available:
        raise ValueError(
            f'num-samples {num_samples} exceeds {available} samples available from '
            f'start-index {start_index}')
    if num_samples == 1:
        return [start_index]
    # Include both ends of the interval. Since num_samples <= available,
    # rounding still yields monotonically increasing, unique indices.
    span = dataset_size - 1 - start_index
    return [round(start_index + i * span / (num_samples - 1))
            for i in range(num_samples)]


def load_checkpoint_strict_except(model, filename, ignored_derived_buffers):
    """Strictly load a checkpoint after dropping known regenerated buffers."""
    from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict

    checkpoint = _load_checkpoint(str(filename), map_location='cpu', logger=None)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    source = checkpoint.get('state_dict', checkpoint)
    metadata = getattr(source, '_metadata', OrderedDict())
    state_dict = OrderedDict(
        (re.sub(r'^module\.', '', key), value) for key, value in source.items())
    state_dict._metadata = metadata

    # Only discard exact, explicitly listed keys that the current model does
    # not persist. All learned parameters and every other mismatch remain
    # covered by strict=True below.
    expected = model.state_dict()
    removed = []
    for key in ignored_derived_buffers:
        if key in state_dict and key not in expected:
            state_dict.pop(key)
            removed.append(key)
    load_state_dict(model, state_dict, strict=True, logger=None)
    return removed


def params(module):
    ps = list(module.parameters())
    return dict(total=sum(p.numel() for p in ps), trainable=sum(p.numel() for p in ps if p.requires_grad),
                parameter_bytes=sum(p.numel() * p.element_size() for p in ps))


def measure(call, warmup, repeat):
    import torch
    import numpy as np
    for _ in range(warmup):
        result = call()
        del result
    torch.cuda.synchronize()
    # Reserved memory after empty_cache is allocator state, not process-wide VRAM.
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    memory = dict(baseline_allocated_mib=base_alloc / 2**20,
                  baseline_reserved_mib=base_reserved / 2**20,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  incremental_peak_allocated_mib=(torch.cuda.max_memory_allocated()-base_alloc) / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    del result
    # Rewarm the allocator so cold allocations from the memory pass do not bias timing.
    result = call()
    del result
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        del result
    return dict(mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p90_ms=float(np.percentile(times,90)), raw_ms=times, **memory)


def heat(feature):
    import torch
    # Native DAOcc fusion tensors are [B,C,X,Y]; plotting requires rows Y, columns X.
    if not torch.isfinite(feature).all():
        raise ValueError('BEV feature contains NaN/Inf')
    return feature[0].float().abs().mean(0).transpose(0,1).cpu().numpy()


def draw_combined(rows, scale, means, covariances, opacities, coverage,
                  destination, xbound, ybound, threshold, max_gaussians=0):
    """Draw feature comparison, Gaussian envelopes and coverage in one figure."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    means = means[0].float().cpu().numpy()
    covariances = covariances[0].float().cpu().numpy()
    opacities = opacities[0, :, 0].float().cpu().numpy()
    active = opacities > threshold
    means, covariances, opacities = means[active], covariances[active], opacities[active]
    total = len(means)
    if max_gaussians and total > max_gaussians:
        selected = np.argpartition(opacities, -max_gaussians)[-max_gaussians:]
        means, covariances, opacities = (value[selected] for value in
                                         (means, covariances, opacities))

    # CUDA adds 0.3 pixel^2 to the projected BEV covariance. Convert that
    # low-pass term back to metres for plotting in the LiDAR XY frame.
    lowpass_x = 0.3 * float(xbound[2]) ** 2
    lowpass_y = 0.3 * float(ybound[2]) ** 2
    angles = np.linspace(0, 2 * np.pi, 33, dtype=np.float32)
    unit_circle = np.stack((np.cos(angles), np.sin(angles)))
    ellipses, valid_opacity = [], []
    for mean, covariance, opacity in zip(means, covariances, opacities):
        sigma = np.array([[covariance[0] + lowpass_x, covariance[1]],
                          [covariance[1], covariance[3] + lowpass_y]],
                         dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
            continue
        contour = mean[:2, None] + eigenvectors @ (
            3.0 * np.sqrt(eigenvalues)[:, None] * unit_circle)
        ellipses.append(contour.T)
        valid_opacity.append(opacity)

    extent = [float(xbound[0]), float(xbound[1]),
              float(ybound[0]), float(ybound[1])]
    fig, plot_axes = plt.subplots(3, 3, figsize=(18, 17))
    feature_axes = plot_axes[:2]
    common = np.percentile(
        np.concatenate([array.ravel() for row in rows for array in row['heats']]),
        [1, 99])
    titles = ('Camera BEV', 'LiDAR BEV (before fusion)',
              'Fused BEV (ConvFuser output)')
    for ri, row in enumerate(rows):
        for ci, (array, title) in enumerate(zip(row['heats'], titles)):
            lo, hi = common if scale == 'shared' else np.percentile(array, [1, 99])
            normalized = np.clip((array - lo) / max(float(hi - lo), 1e-12), 0, 1)
            axis = feature_axes[ri][ci]
            axis.imshow(normalized, origin='lower', extent=extent,
                        cmap='viridis', vmin=0, vmax=1)
            axis.set_title(row['name'] + ' | ' + title +
                           (' [random init]' if row['random'] else ''), fontsize=9)
            axis.set_xlabel('LiDAR X (m)'); axis.set_ylabel('LiDAR Y (m)')
            row['scales'].append([float(lo), float(hi)])

    axes = plot_axes[2, :2]
    plot_axes[2, 2].axis('off')
    if ellipses:
        collection = LineCollection(ellipses, cmap='viridis', linewidths=.25,
                                    alpha=.55, rasterized=True)
        collection.set_array(np.asarray(valid_opacity))
        collection.set_clim(0, 1)
        axes[0].add_collection(collection)
        fig.colorbar(collection, ax=axes[0], fraction=.046, pad=.04, label='Opacity')
    axes[0].set_xlim(extent[:2]); axes[0].set_ylim(extent[2:])
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_title(f'Projected 3σ Gaussian envelopes: {len(ellipses)}/{total}')
    axes[0].set_xlabel('LiDAR X (m)'); axes[0].set_ylabel('LiDAR Y (m)')
    # Draw the exact renderer cell boundaries. For the current configuration
    # this is a 180x180 grid over [-54, 54] m with 0.6 m cells. Minor tick
    # labels/marks stay hidden so the dense grid does not clutter the axes.
    x_edges = np.arange(float(xbound[0]), float(xbound[1]) +
                        float(xbound[2]) * .5, float(xbound[2]))
    y_edges = np.arange(float(ybound[0]), float(ybound[1]) +
                        float(ybound[2]) * .5, float(ybound[2]))
    axes[0].set_xticks(x_edges, minor=True)
    axes[0].set_yticks(y_edges, minor=True)
    axes[0].tick_params(which='minor', bottom=False, left=False)
    axes[0].grid(which='minor', color='0.52', linewidth=.28, alpha=.45)
    axes[0].grid(which='major', color='0.45', linewidth=.45, alpha=.35)

    image = axes[1].imshow(coverage, origin='lower', extent=extent,
                           cmap='magma', vmin=0, vmax=1)
    fig.colorbar(image, ax=axes[1], fraction=.046, pad=.04,
                 label='Accumulated alpha (1 - transmittance)')
    axes[1].set_title('Exact Gaussian rasterization coverage')
    axes[1].set_xlabel('LiDAR X (m)'); axes[1].set_ylabel('LiDAR Y (m)')
    fig.suptitle('BEV feature comparison and GaussianBEV geometry/coverage\n'
                 'Features: mean absolute channel response, not occupancy probability; '
                 f'color range: {scale}; visualization excluded from timing')
    fig.subplots_adjust(left=.055, right=.945, bottom=.045, top=.925,
                        wspace=.38, hspace=.34)
    fig.savefig(destination, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return dict(active_gaussians=int(total), drawn_gaussians=int(len(ellipses)),
                max_coverage=float(np.max(coverage)),
                mean_coverage=float(np.mean(coverage)))


def main():
    args = arguments()
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from torchpack.utils.config import Config as TorchConfig
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    # Explicit import also supports older checkouts without the normal registration import.
    import mmdet3d.models.vtransforms.gaussian_bev

    def seed():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    torch.cuda.set_device(args.gpu)
    configs=[]
    with suppress_external_output():
        for path in (args.baseline_config,args.gaussian_config):
            raw=TorchConfig(); raw.load(str(path.resolve()),recursive=True)
            configs.append(Config(recursive_eval(raw)))
    for key in ('type','ann_file','pipeline'):
        if configs[0].data.val.get(key) != configs[1].data.val.get(key):
            raise ValueError('Validation data mismatch: '+key)
    if configs[0].point_cloud_range != configs[1].point_cloud_range:
        raise ValueError('Different LiDAR ranges; cannot compare aligned native BEV maps')
    expected=('BEVTransformV2','GaussianBEVTransform')
    for cfg,typ in zip(configs,expected):
        if cfg.model.encoders.camera.vtransform.type != typ:
            raise ValueError('Expected vtransform '+typ)
    data_cfg=deepcopy(configs[0].data.val)
    data_cfg.test_mode=True; data_cfg.pop('samples_per_gpu',None)
    with suppress_external_output():
        dataset=build_dataset(data_cfg)
    indices=uniformly_spaced_indices(len(dataset), args.start_index, args.num_samples)
    seed()
    # Cache identical CPU inputs once; both methods see precisely the same frames and transforms.
    batches=[collate([dataset[i]],samples_per_gpu=1) for i in indices]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(precision=args.precision,batch_size=1,seed=args.seed,gpu=torch.cuda.get_device_name(args.gpu),
                indices=indices,warmup=args.warmup,repeat=args.repeat,methods={},
                performance_measured=args.measure_performance,
                timing=('Synchronized wall time; inference only; data loading/transfers/plotting excluded.'
                        if args.measure_performance else 'Disabled by --skip-performance.'),
                stages={'vtransform':'FPN feature to camera BEV only',
                        'camera_total':'image backbone + FPN + vtransform',
                        'lidar':'voxelization + LiDAR backbone',
                        'fusion':'ConvFuser only'},
                memory=('PyTorch allocator MiB, separate untimed memory pass; delta includes output and temporary tensors. '
                        'Baseline includes resident model and inputs. Does not include all CUDA/context/external allocations.'
                        if args.measure_performance else 'Disabled by --skip-performance.'),
                visualization='native LiDAR frame; X horizontal, Y vertical; mean(abs(channels)); fuser output before decoder',
                note='No checkpoints means seeded random initialization, not trained semantic features. '
                     'Gaussian opacity/radius distributions can change runtime after training.')
    cached={}; csv_rows=[]
    names=('BEVTransformV2','GaussianBEV')
    checkpoints=(args.baseline_checkpoint,args.gaussian_checkpoint)
    gaussian_derived_buffers=(
        'encoders.camera.vtransform.geometry.covariance',
        'encoders.camera.vtransform.renderer.view',
        'encoders.camera.vtransform.renderer.projection',
    )
    with torch.no_grad():
        for name,cfg,checkpoint in zip(names,configs,checkpoints):
            seed()
            mc=deepcopy(cfg.model); clear_pretrained(mc)
            ignored_checkpoint_keys=[]
            with suppress_external_output():
                model=build_model(mc)
                if checkpoint:
                    if name == 'GaussianBEV':
                        ignored_checkpoint_keys=load_checkpoint_strict_except(
                            model, checkpoint, gaussian_derived_buffers)
                    else:
                        load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
                else:
                    model.init_weights()
            if ignored_checkpoint_keys:
                print('GaussianBEV checkpoint: ignored regenerated buffers: ' +
                      ', '.join(ignored_checkpoint_keys), flush=True)
            model.float().eval()
            for module in model.modules():
                if hasattr(module,'fp16_enabled'): module.fp16_enabled=False
            if args.precision=='fp16': wrap_fp16_model(model)
            parameter_counts={'model':params(model),'camera':params(model.encoders['camera']),
                              'vtransform':params(model.encoders['camera']['vtransform']),
                              'lidar':params(model.encoders['lidar']),'fuser':params(model.fuser)}
            model.cuda(args.gpu)
            results=[]
            for index,batch in zip(indices,batches):
                data=scatter(batch,[args.gpu])[0]
                if not torch.is_tensor(data['img']):
                    raise ValueError('Test-time augmentation lists are not supported; use standard validation pipeline')
                keys=('points','camera2ego','lidar2ego','lidar2camera','lidar2image',
                      'camera_intrinsics','camera2lidar','img_aug_matrix','lidar_aug_matrix')
                metadata=data.get('metas',data.get('img_metas'))
                vt_args=[data[k] for k in keys]+[metadata]
                extra={k:data[k] for k in ('camera_ego2global',) if k in data}
                def camera():
                    images=data['img']
                    if args.precision=='fp16': images=images.half()
                    return model.extract_camera_features(images,*vt_args,**extra)
                frame=dict(index=index,token=str(dataset.data_infos[index].get('token',index)),stages={})
                if args.measure_performance:
                    frame['stages']['camera_total']=measure(camera,args.warmup,args.repeat)
                    # Precompute FPN once; exclude it from view-transform-only timing/memory delta.
                    images=data['img']; b,n,c,h,w=images.shape
                    images=images.reshape(b*n,c,h,w)
                    if args.precision=='fp16': images=images.half()
                    fp=model.encoders['camera']['neck'](model.encoders['camera']['backbone'](images))
                    if isinstance(fp,(list,tuple)): fp=fp[0]
                    fp=fp.reshape(b,n,*fp.shape[1:])
                    frame['stages']['vtransform']=measure(
                        lambda:model.encoders['camera']['vtransform'](fp,*vt_args,**extra),
                        args.warmup,args.repeat)
                    del fp,images
                    frame['stages']['lidar']=measure(
                        lambda:model.extract_lidar_features(data['points']),
                        args.warmup,args.repeat)
                # Capture the actual tensors entering the Gaussian renderer only
                # for the final, untimed visualization forward. Timed calls above
                # remain unchanged and do not include hooks or extra rendering.
                gaussian_inputs = None
                if name == names[1]:
                    def capture_gaussians(_module, module_inputs):
                        nonlocal gaussian_inputs
                        gaussian_inputs = tuple(module_inputs[:4])
                    hook = model.encoders['camera']['vtransform'].renderer.register_forward_pre_hook(
                        capture_gaussians)
                    try:
                        camera_bev = camera()
                    finally:
                        hook.remove()
                    if gaussian_inputs is None:
                        raise RuntimeError('Failed to capture Gaussian renderer inputs')
                else:
                    camera_bev=camera()
                lidar_bev=model.extract_lidar_features(data['points'])
                if camera_bev.shape[-2:] != lidar_bev.shape[-2:]:
                    raise ValueError('Camera/LiDAR BEV shapes do not align')
                # Existing BEVFusion establishes camera then lidar encoder order.
                inputs=[{'camera':camera_bev,'lidar':lidar_bev}[key] for key in model.encoders]
                if args.measure_performance:
                    frame['stages']['fusion']=measure(
                        lambda:model.fuser(inputs),args.warmup,args.repeat)
                fused=model.fuser(inputs)
                frame['shapes']={k:list(v.shape) for k,v in [('camera',camera_bev),('lidar',lidar_bev),('fused',fused)]}
                heats=[heat(v) for v in (camera_bev,lidar_bev,fused)]
                cached[(name,index)]=dict(name=name,random=checkpoint is None,heats=heats,scales=[])
                if name == names[1]:
                    bounds=configs[0].point_cloud_range
                    extent=[bounds[0],bounds[3],bounds[1],bounds[4]]
                    rows=[cached[(method,index)] for method in names]
                    path=args.output_dir/f'comparison_{index:06d}.png'
                    # Rendering constant-one features with the exact captured
                    # geometry produces accumulated alpha (1-transmittance), i.e.
                    # the renderer's actual coverage rather than a feature proxy.
                    gaussian_features, gaussian_means, gaussian_cov, gaussian_opacity = gaussian_inputs
                    renderer = model.encoders['camera']['vtransform'].renderer
                    coverage_bev = renderer(
                        torch.ones_like(gaussian_features), gaussian_means,
                        gaussian_cov, gaussian_opacity)
                    coverage = coverage_bev[0, 0].float().transpose(0, 1).cpu().numpy()
                    vcfg = cfg.model.encoders.camera.vtransform
                    gaussian_stats = draw_combined(
                        rows, args.feature_scale, gaussian_means, gaussian_cov,
                        gaussian_opacity, coverage, path, vcfg.xbound, vcfg.ybound,
                        renderer.threshold, args.max_gaussians_vis)
                    report.setdefault('plots',[]).append(dict(
                        index=index, path=str(path), extent=extent,
                        color_limits={row['name']:row['scales'] for row in rows}))
                    report.setdefault('gaussian_plots', []).append(
                        dict(index=index, path=str(path), **gaussian_stats))
                    print('Saved combined visualization:',path,flush=True)
                    del coverage_bev, gaussian_inputs
                    for method in names:
                        del cached[(method,index)]
                results.append(frame)
                for stage,values in frame['stages'].items():
                    csv_rows.append(dict(method=name,index=index,stage=stage,**{k:v for k,v in values.items() if k!='raw_ms'}))
                if args.measure_performance:
                    print(f'{name} [{index}]: camera={frame["stages"]["camera_total"]["mean_ms"]:.2f} ms, '
                          f'vtransform={frame["stages"]["vtransform"]["mean_ms"]:.2f} ms',flush=True)
                else:
                    print(f'{name} [{index}]: visualization forward complete',flush=True)
                del data,vt_args,extra,inputs,camera_bev,lidar_bev,fused
            summary={}
            if args.measure_performance:
                for stage in report['stages']:
                    times=[v for frame in results for v in frame['stages'][stage]['raw_ms']]
                    summary[stage]=dict(mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
                                        p90_ms=float(np.percentile(times,90)),
                                        max_incremental_peak_allocated_mib=max(f['stages'][stage]['incremental_peak_allocated_mib'] for f in results))
            report['methods'][name]=dict(config=str(Path(cfg.filename).resolve()) if cfg.filename else str(args.baseline_config if name==names[0] else args.gaussian_config),
                                         checkpoint=str(checkpoint) if checkpoint else None,parameters=parameter_counts,
                                         summary=summary,frames=results)
            (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            del model
            gc.collect(); torch.cuda.empty_cache()
    measurement_fields=('method','index','stage','mean_ms','median_ms','p90_ms',
                        'baseline_allocated_mib','baseline_reserved_mib',
                        'peak_allocated_mib','incremental_peak_allocated_mib',
                        'peak_reserved_mib')
    with (args.output_dir/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=measurement_fields)
        writer.writeheader(); writer.writerows(csv_rows)
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Saved reports and combined visualization PNGs to',args.output_dir)
    for name in names:
        m=report['methods'][name]
        message=(name+' vtransform parameters: '+str(m['parameters']['vtransform']['total'])+
                 ' camera parameters: '+str(m['parameters']['camera']['total']))
        if args.measure_performance:
            message += ' vtransform mean ms: '+str(round(m['summary']['vtransform']['mean_ms'],3))
        print(message)


if __name__=='__main__':
    main()
