#!/usr/bin/env python3
"""单卡测量实例注入的推理开销；只新增脚本，不修改模型源码。

先完整测关闭注入，再完整测开启注入；分别预热，使用相同权重、样本顺序及随机种子。
统计实例注入新增参数和完整推理耗时，不执行分项计时插桩。
"""
import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    from mmcv import DictAction
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('checkpoint', nargs='?', help='可选：实例融合模型权重；省略则随机初始化')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--warmup', type=int, default=10, help='两种模式分别预热的 batch 数')
    p.add_argument('--steps', type=int, default=50, help='正式计时 batch 数')
    p.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    #* (11.16) 兼容旧命令，但不再执行分项计时。
    p.add_argument('--profile', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--params-only', action='store_true', help='仅构建模型统计参数，不加载数据、不运行 GPU 推理')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()
    if min(args.batch_size, args.steps, args.warmup) < 1 or args.workers < 0:
        p.error('batch-size、steps、warmup 必须为正，workers 不能为负')
    return args


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), std=float(a.std()))


def measured(call):
    import torch
    # 包含 CPU 候选筛选/NMS 调度及 GPU 计算；不只测 CUDA kernel 时间。
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def parameter_report(model):
    """只把注入专用参数计为新增；检测 shared_conv/原任务头不重复计入。"""
    def unique(modules):
        return {id(p): p for module in modules if module is not None for p in module.parameters()}

    def count(parameters):
        return dict(total=sum(p.numel() for p in parameters.values()),
                    trainable=sum(p.numel() for p in parameters.values() if p.requires_grad))

    detection, occupancy = model.heads['object'], model.heads['occ']
    fusion = occupancy.instance_fusion
    predictor = getattr(detection, 'offset_predictor', None)
    weight_predictor = getattr(detection, 'sample_weight_predictor', None)
    components = {'sampling_offset_predictor': count(unique([predictor])),
                  'sampling_weight_predictor': count(unique([weight_predictor]))}
    for name, module in fusion.named_children():
        components['instance_fusion.' + name] = count(unique([module]))
    all_params = unique([model])
    additional = unique([fusion, predictor, weight_predictor])
    remainder = {key: value for key, value in all_params.items() if key not in additional}
    total, added, base = count(all_params), count(additional), count(remainder)
    return dict(
        model=total, without_instance_modules=base, instance_modules=added,
        additional_million=added['total'] / 1e6,
        increase_percent=100 * added['total'] / base['total'] if base['total'] else None,
        components=components,
        options=dict(roi_features=getattr(detection, 'roi_features', False),
                     roi_output_size=getattr(detection, 'roi_output_size', None),
                     roi_sampling_ratio=getattr(detection, 'roi_sampling_ratio', None),
                     multi_point_features=getattr(detection, 'multi_point_features', False),
                     learnable_sampling=predictor is not None,
                     sampling_points=(getattr(detection, 'sampling_points', 5) if predictor is not None
                                      else (5 if getattr(detection, 'multi_point_features', False) else 0)),
                     edge_matching=getattr(fusion, 'edge_match', None) is not None,
                     local_edges=getattr(fusion, 'local_edges', False)),
        note='Counts unique Parameter elements, excluding buffers. Added = BoxBEVFusion + offset/weight predictors; '
             'original detection shared_conv and task heads are reused. Local edge enumeration and '
             'fixed sampling have no parameters. Off/on retains the same allocated model; subtraction '
             'is architectural accounting, not a second model or GPU memory measurement.')


def print_parameters(report):
    print('实例注入新增参数（不含复用的原检测头）：', flush=True)
    for name, values in report['components'].items():
        print(f"  {name}: {values['total']:,}")
    print(f"新增合计：{report['instance_modules']['total']:,} "
          f"({report['additional_million']:.6f} M)；"
          f"其中可训练：{report['instance_modules']['trainable']:,}")
    print(f"完整模型：{report['model']['total']:,}；"
          f"扣除注入模块：{report['without_instance_modules']['total']:,}")
    print('实际开关：' + json.dumps(report['options'], ensure_ascii=False), flush=True)


def main():
    args = parse_args()
    import torch
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from torchpack.utils.config import Config as TorchConfig
    from mmdet.apis import set_random_seed
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.models.fusion_models.instance_bevfusion import InstanceBEVFusion
    from mmdet3d.utils import recursive_eval

    if not args.params_only:
        if not torch.cuda.is_available():
            raise RuntimeError('需要一张可用 CUDA GPU，或使用 --params-only')
        torch.cuda.set_device(0)
    set_random_seed(args.seed)
    raw = TorchConfig()
    raw.load(str(Path(args.config).resolve()), recursive=True)
    cfg = Config(recursive_eval(raw))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    mc = deepcopy(cfg.model)
    mc.train_cfg = None
    model = build_model(mc, test_cfg=cfg.get('test_cfg'))
    if not isinstance(model, InstanceBEVFusion):
        raise TypeError('需要 InstanceBEVFusion 配置')
    #* (11.19) 按当前开关统计实际构建的参数；不把复用检测网络当作新增参数。
    parameters = parameter_report(model)
    print_parameters(parameters)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(args.out_dir / 'resolved_config.py'))
    (args.out_dir / 'parameters.json').write_text(json.dumps(parameters, indent=2, ensure_ascii=False))
    if args.params_only:
        print(f'参数统计保存至：{args.out_dir}；未运行推理。', flush=True)
        return
    dc = deepcopy(cfg.data.val)
    dc.pop('samples_per_gpu', None)
    dc.test_mode = True
    if args.batch_size > 1:
        dc.pipeline = replace_ImageToTensor(dc.pipeline)
    dataset = build_dataset(dc)
    # 只测完整 batch，避免末尾小 batch 混入统计；不循环重复数据填充次数。
    if (args.warmup + args.steps) * args.batch_size > len(dataset):
        raise ValueError('验证集不足以提供指定数量的完整预热和计时 batch')
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    else:
        print('未加载完整任务权重：沿用模型构造时初始化（可含 backbone 预训练）；候选分布不代表训练后模型。', flush=True)
    model.float().cuda().eval()
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    if args.precision == 'fp16':
        wrap_fp16_model(model)
    rows = [dict(step=i + 1, batch_size=args.batch_size) for i in range(args.steps)]
    original_flag = model.instance_to_occ
    print('完整推理计时：不含数据加载/输入搬运；含检测后处理和预测输出 CPU 拷贝。', flush=True)
    if args.profile:
        print('--profile 已停用，仅测整体推理耗时。', flush=True)
    #* (11.16) 两个完整阶段各自预热，不交替切换，也不额外执行分项前向。
    # 每阶段重建顺序 loader 并重置随机种子，使两阶段读取相同验证样本。
    try:
        with torch.no_grad():
            for enabled, key, label in [(False, 'off_ms', '关闭注入'),
                                        (True, 'on_ms', '开启注入')]:
                model.instance_to_occ = enabled
                set_random_seed(args.seed)
                loader = build_dataloader(
                    dataset, samples_per_gpu=args.batch_size,
                    workers_per_gpu=args.workers, num_gpus=1,
                    dist=False, shuffle=False, seed=args.seed)
                iterator = iter(loader)
                print(f'{label}：预热 {args.warmup} 步，再计时 {args.steps} 步。', flush=True)
                for i in range(args.warmup + args.steps):
                    batch = next(iterator)
                    data = scatter(batch, [0])[0]
                    output, elapsed = measured(lambda: model(**data))
                    if len(output) != args.batch_size:
                        raise RuntimeError('预测数量与 batch-size 不一致')
                    del output, data, batch
                    if i < args.warmup:
                        continue
                    step = i - args.warmup
                    rows[step][key] = elapsed
                    if (step + 1) % 10 == 0 or step + 1 == args.steps:
                        print(f'{label} {step + 1}/{args.steps}: {elapsed:.2f} ms', flush=True)
                del iterator, loader
    finally:
        model.instance_to_occ = original_flag
    for row in rows:
        row['delta_ms'] = row['on_ms'] - row['off_ms']

    off = stats([r['off_ms'] for r in rows])
    on = stats([r['on_ms'] for r in rows])
    result = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
                  random_weights=args.checkpoint is None,
                  parameters=parameters,
                  precision=args.precision, gpu=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, seed=args.seed, batch_size=args.batch_size,
                  warmup=args.warmup, steps=len(rows), off_ms_per_batch=off, on_ms_per_batch=on,
                  measurement_order=['off', 'on'],
                  delta_note='Same batch index across separate phases, not adjacent timed forwards.',
                  paired_delta_ms_per_batch=stats([r['delta_ms'] for r in rows]),
                  overhead_percent=(on['mean'] / off['mean'] - 1) * 100,
                  off_ms_per_frame=off['mean'] / args.batch_size,
                  on_ms_per_frame=on['mean'] / args.batch_size,
                  off_frames_per_second=1000 * args.batch_size / off['mean'],
                  on_frames_per_second=1000 * args.batch_size / on['mean'],
                  scope='eval/no_grad; excludes data loading and input transfer; includes output CPU transfer; '
                        'same InstanceBEVFusion weights with instance_to_occ off/on, not a separate baseline checkpoint')
    with (args.out_dir / 'timings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'中位数：关闭 {off["median"]:.2f} / 开启 {on["median"]:.2f} ms/batch；'
          f'P90：关闭 {off["p90"]:.2f} / 开启 {on["p90"]:.2f} ms/batch')
    print(f'关闭注入：{off["mean"]:.2f} ms/batch；开启注入：{on["mean"]:.2f} ms/batch\n'
          f'新增：{on["mean"]-off["mean"]:.2f} ms/batch '
          f'({result["overhead_percent"]:+.2f}%)\n保存至：{args.out_dir}')


if __name__ == '__main__':
    main()
