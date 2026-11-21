#!/usr/bin/env python3
"""同帧比较 BEVTransformV2 / GaussianBEVTransform 的推理开销及 BEV 特征。

从 DAOCC 根目录运行（两份 checkpoint 均可省略，省略时明确标记随机初始化）：
python tools/compare_gaussian_bev.py --output-dir outputs/bev_compare \
    --baseline-checkpoint BASE.pth --gaussian-checkpoint GAUSSIAN.pth

默认 FP32、batch=1；只执行编码和 ConvFuser，不执行 decoder/任务头。
计时含 CPU 调度和 GPU 执行，不含加载、数据搬运、可视化。
"""
import argparse
import csv
import gc
import os
import json
import logging
import random
import sys
import time
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
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5, help='每个样本每个测量阶段预热次数')
    p.add_argument('--repeat', type=int, default=10, help='每个样本每阶段计时次数')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--feature-scale', choices=['independent', 'shared'], default='independent')
    a = p.parse_args()
    if a.num_samples < 1 or a.repeat < 1 or a.warmup < 0 or a.start_index < 0:
        p.error('num-samples/repeat必须为正；warmup/start-index不能为负')
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


def draw(rows, destination, scale, extent):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    common = np.percentile(np.concatenate([a.ravel() for row in rows for a in row['heats']]), [1,99])
    fig, axes = plt.subplots(2,3,figsize=(15,10))
    for ri,row in enumerate(rows):
        for ci,(array,title) in enumerate(zip(row['heats'],['Camera BEV','LiDAR BEV (before fusion)','Fused BEV (ConvFuser output)'])):
            lo,hi = common if scale == 'shared' else np.percentile(array,[1,99])
            normalized = np.clip((array-lo)/max(float(hi-lo),1e-12),0,1)
            axes[ri,ci].imshow(normalized,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=1)
            axes[ri,ci].set_title(row['name']+' | '+title+(' [random init]' if row['random'] else ''),fontsize=9)
            axes[ri,ci].set_xlabel('LiDAR X (m)'); axes[ri,ci].set_ylabel('LiDAR Y (m)')
            row['scales'].append([float(lo),float(hi)])
    fig.suptitle('Mean absolute channel response; not occupancy probability. Color range: '+scale)
    fig.tight_layout()
    fig.savefig(destination,dpi=160)
    plt.close(fig)


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
    stop=args.start_index+args.num_samples
    if stop>len(dataset):
        raise ValueError('Requested sample range exceeds validation dataset')
    indices=list(range(args.start_index,stop))
    seed()
    # Cache identical CPU inputs once; both methods see precisely the same frames and transforms.
    batches=[collate([dataset[i]],samples_per_gpu=1) for i in indices]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(precision=args.precision,batch_size=1,seed=args.seed,gpu=torch.cuda.get_device_name(args.gpu),
                indices=indices,warmup=args.warmup,repeat=args.repeat,methods={},
                timing='Synchronized wall time; inference only; data loading/transfers/plotting excluded.',
                stages={'vtransform':'FPN feature to camera BEV only',
                        'camera_total':'image backbone + FPN + vtransform',
                        'lidar':'voxelization + LiDAR backbone',
                        'fusion':'ConvFuser only'},
                memory='PyTorch allocator MiB, separate untimed memory pass; delta includes output and temporary tensors. '
                       'Baseline includes resident model and inputs. Does not include all CUDA/context/external allocations.',
                visualization='native LiDAR frame; X horizontal, Y vertical; mean(abs(channels)); fuser output before decoder',
                note='No checkpoints means seeded random initialization, not trained semantic features. '
                     'Gaussian opacity/radius distributions can change runtime after training.')
    cached={}; csv_rows=[]
    names=('BEVTransformV2','GaussianBEV')
    checkpoints=(args.baseline_checkpoint,args.gaussian_checkpoint)
    with torch.no_grad():
        for name,cfg,checkpoint in zip(names,configs,checkpoints):
            seed()
            mc=deepcopy(cfg.model); clear_pretrained(mc)
            with suppress_external_output():
                model=build_model(mc)
                if checkpoint:
                    load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
                else:
                    model.init_weights()
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
                frame['stages']['camera_total']=measure(camera,args.warmup,args.repeat)
                # Precompute FPN once; exclude it from view-transform-only timing/memory delta.
                images=data['img']; b,n,c,h,w=images.shape
                images=images.reshape(b*n,c,h,w)
                if args.precision=='fp16': images=images.half()
                fp=model.encoders['camera']['neck'](model.encoders['camera']['backbone'](images))
                if isinstance(fp,(list,tuple)): fp=fp[0]
                fp=fp.reshape(b,n,*fp.shape[1:])
                frame['stages']['vtransform']=measure(
                    lambda:model.encoders['camera']['vtransform'](fp,*vt_args,**extra),args.warmup,args.repeat)
                del fp,images
                frame['stages']['lidar']=measure(lambda:model.extract_lidar_features(data['points']),args.warmup,args.repeat)
                camera_bev=camera()
                lidar_bev=model.extract_lidar_features(data['points'])
                if camera_bev.shape[-2:] != lidar_bev.shape[-2:]:
                    raise ValueError('Camera/LiDAR BEV shapes do not align')
                # Existing BEVFusion establishes camera then lidar encoder order.
                inputs=[{'camera':camera_bev,'lidar':lidar_bev}[key] for key in model.encoders]
                frame['stages']['fusion']=measure(lambda:model.fuser(inputs),args.warmup,args.repeat)
                fused=model.fuser(inputs)
                frame['shapes']={k:list(v.shape) for k,v in [('camera',camera_bev),('lidar',lidar_bev),('fused',fused)]}
                heats=[heat(v) for v in (camera_bev,lidar_bev,fused)]
                cached[(name,index)]=dict(name=name,random=checkpoint is None,heats=heats,scales=[])
                if name == names[1]:
                    bounds=configs[0].point_cloud_range
                    extent=[bounds[0],bounds[3],bounds[1],bounds[4]]
                    rows=[cached[(method,index)] for method in names]
                    path=args.output_dir/f'comparison_{index:06d}.png'
                    draw(rows,path,args.feature_scale,extent)
                    report.setdefault('plots',[]).append(dict(index=index,path=str(path),extent=extent,
                        color_limits={row['name']:row['scales'] for row in rows}))
                    print('Saved image:',path,flush=True)
                    for method in names:
                        del cached[(method,index)]
                results.append(frame)
                for stage,values in frame['stages'].items():
                    csv_rows.append(dict(method=name,index=index,stage=stage,**{k:v for k,v in values.items() if k!='raw_ms'}))
                print(f'{name} [{index}]: camera={frame["stages"]["camera_total"]["mean_ms"]:.2f} ms, '
                      f'vtransform={frame["stages"]["vtransform"]["mean_ms"]:.2f} ms',flush=True)
                del data,vt_args,extra,inputs,camera_bev,lidar_bev,fused
            summary={}
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
    with (args.output_dir/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Saved report.json, measurements.csv and comparison PNGs to',args.output_dir)
    for name in names:
        m=report['methods'][name]
        print(name,'vtransform parameters:',m['parameters']['vtransform']['total'],
              'camera parameters:',m['parameters']['camera']['total'],
              'vtransform mean ms:',round(m['summary']['vtransform']['mean_ms'],3))


if __name__=='__main__':
    main()
