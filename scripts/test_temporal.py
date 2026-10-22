"""用 CPU 小张量检查 BEV 对齐和融合；不加载数据集、模型权重或 CUDA 扩展。"""
import torch
import sys
from pathlib import Path

# ! scripts 的上一级是项目根目录；普通导入纯 PyTorch 组件，不加载完整模型。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.TemporalDAOcc.fusion import TemporalFusion, warp_bev


# *=================== 1. 检查 BEV 对齐 ===================
def check_bev_alignment():
    # 输入：[1 个样本，1 个通道，X 方向 4 格，Y 方向 6 格]。
    # 每格数值不同，方便判断对齐后究竟取到了哪个位置。
    previous = torch.arange(24.).reshape(1, 1, 4, 6)
    bounds = [0., 0., 4., 6.]
    transform = torch.eye(4).unsqueeze(0)

    # 两帧坐标一致：调用 fusion.py 的 warp_bev，结果应保持不变。
    aligned, valid = warp_bev(previous, transform, bounds)
    torch.testing.assert_close(aligned, previous)
    assert valid.all(), "坐标一致时，所有位置都应有有效历史"

    # 当前位置映射到历史时 X 加 1：应读取历史的下一个 X 格点。
    transform[:, 0, 3] = 1.
    aligned, valid = warp_bev(previous, transform, bounds)
    torch.testing.assert_close(aligned[:, :, :-1], previous[:, :, 1:])
    assert not valid[:, :, -1].any(), "最后一行越界，必须屏蔽"
    print("  通过：恒等对齐、平移方向、越界掩码")


# *=================== 2. 检查融合结果与无历史回退 ===================
def check_feature_fusion():
    current = torch.ones(2, 3, 4, 6)
    history = torch.full_like(current, 3.)
    valid = torch.ones(2, 1, 4, 6, dtype=torch.bool)
    valid[0] = False  # 第一个样本没有历史，第二个样本有历史。

    # 直接调用融合模块的 forward：有效位置各取一半，应得到 (1+3)/2=2。
    layer = TemporalFusion(channels=3, mode='fixed', alpha=.5)
    output = layer(current, history, valid)
    assert torch.equal(output[0], current[0]), "无历史时必须保持当前特征"
    torch.testing.assert_close(output[1], torch.full_like(output[1], 2.))

    # 关闭融合后，不论历史是否有效，都必须保持当前特征。
    disabled = TemporalFusion(channels=3, mode='off')
    assert torch.equal(disabled(current, history, valid), current)
    print("  通过：固定权重融合、无效历史回退、关闭融合")


# *=================== 3. 检查可学习门控能否反传梯度 ===================
def check_fusion_gradients():
    current = torch.ones(1, 3, 4, 6, requires_grad=True)
    history = torch.full_like(current, 3.)
    valid = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    layer = TemporalFusion(channels=3, mode='gate')

    output = layer(current, history, valid)
    # 将输出求和作为简单测试损失，调用 backward 检查反传链路。
    # 这里只检查梯度，不代表门控已经训练好，更不代表预测精度提升。
    loss = output.sum()
    loss.backward()
    assert current.grad is not None, "当前特征应收到梯度"
    assert layer.gate.weight.grad is not None, "门控卷积应收到梯度"
    assert torch.isfinite(layer.gate.weight.grad).all(), "梯度不能含 NaN/Inf"
    assert layer.gate.weight.grad.abs().sum() > 0, "本例门控权重梯度应非零"
    print("  通过：当前特征和门控参数均可反传")


# *=================== 4. 检查位姿与数据增强的坐标关系 ===================
def check_pose_alignment():
    previous = torch.arange(24.).reshape(1, 1, 4, 6)
    bounds = [0., 0., 4., 6.]
    previous_pose = torch.eye(4).unsqueeze(0)
    current_pose = torch.eye(4).unsqueeze(0)
    current_pose[:, 0, 3] = 1.  # 当前 LiDAR 相对历史沿世界 X 方向移动 1。
    previous_aug = torch.eye(4).unsqueeze(0)
    current_aug = torch.eye(4).unsqueeze(0)

    # * 先撤销当前增强，再转世界坐标，再转历史 LiDAR 坐标，最后施加历史增强。
    # 本例直接展示公式并验证采样结果，不调用完整模型的缓存接口。
    transform = (previous_aug @ torch.inverse(previous_pose)
                 @ current_pose @ torch.inverse(current_aug))
    aligned, valid = warp_bev(previous, transform, bounds)
    torch.testing.assert_close(aligned[:, :, :-1], previous[:, :, 1:])

    # 当前增强又沿 X 平移 1，撤销增强后恰好抵消位姿平移，应恢复恒等采样。
    current_aug[:, 0, 3] = 1.
    transform = (previous_aug @ torch.inverse(previous_pose)
                 @ current_pose @ torch.inverse(current_aug))
    aligned, valid = warp_bev(previous, transform, bounds)
    torch.testing.assert_close(aligned, previous)
    assert valid.all()
    print("  通过：位姿平移与增强矩阵的方向关系")


# *=================== 5. 绘制当前、历史、对齐历史的三联图 ===================
def save_bev_comparison(current, history, aligned, valid, bounds, path):
    import numpy as np
    import matplotlib
    # 无窗口保存，避免 WSL 中 Qt/xcb 冲突。
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    # 与 visualize-fig5.py 一样，先对通道取绝对值均值。
    # 原始张量为 [B,C,X,Y]；画图转成 [Y,X]，横轴 X、纵轴 Y。
    heats = [feature[0].detach().float().abs().mean(0).cpu().numpy().T
             for feature in (current, history, aligned)]
    mask = valid[0, 0].detach().cpu().numpy().T
    if not all(np.isfinite(heat).all() for heat in heats):
        raise ValueError('BEV 特征包含 NaN/Inf')
    # * 三幅图共用 1%/99% 分位色标，避免各自归一化掩盖幅值差异。
    # 对齐后无效位置不参与色标统计，也不画成低响应区域。
    values = np.concatenate([heats[0].ravel(), heats[1].ravel(), heats[2][mask]])
    low, high = [float(v) for v in np.percentile(values, [1, 99])]
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')
    xmin, ymin, xmax, ymax = bounds
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    titles = ['(a) Current BEV', '(b) Previous BEV', '(c) Aligned Previous BEV']
    try:
        for i, (ax, heat, title) in enumerate(zip(axes, heats, titles)):
            normalized = np.clip((heat-low)/max(high-low, 1e-12), 0, 1)
            if i == 2:
                normalized = np.ma.masked_where(~mask, normalized)
            ax.imshow(normalized, origin='lower', extent=(xmin, xmax, ymin, ymax),
                      cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_aspect('equal')
            ax.grid(False, which='both')
            ax.set_axis_off()
            ax.text(.5, -.025, title, transform=ax.transAxes, ha='center', va='top',
                    fontsize=16, fontfamily='serif')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.04, facecolor='white')
    finally:
        plt.close(fig)
    return low, high, float(mask.mean())


#! 每行一帧历史，列为原始历史、对齐历史、当前帧参照；行顺序由调用者按时间排好。
def save_history_sequence(current_heat, panels, bounds, path, current_number):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    values = [current_heat.ravel()]
    for original, aligned, mask, label in panels:
        if original is not None:
            values.extend([original.ravel(), aligned[mask]])
    values = np.concatenate(values)
    if not np.isfinite(values).all():
        raise ValueError('BEV 热力图包含 NaN/Inf')
    low, high = [float(v) for v in np.percentile(values, [1, 99])]
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')
    xmin, ymin, xmax, ymax = bounds
    fig, axes = plt.subplots(len(panels), 3, figsize=(13, 4.5*len(panels)), squeeze=False)
    try:
        for row, (original, aligned, mask, label) in enumerate(panels):
            titles = [label+' / Original', label+' / Aligned', f'Current frame {current_number}']
            for col, heat in enumerate([original, aligned, current_heat]):
                ax = axes[row, col]
                if heat is not None:
                    normalized = np.clip((heat-low)/max(high-low, 1e-12), 0, 1)
                    if col == 1:
                        normalized = np.ma.masked_where(~mask, normalized)
                    ax.imshow(normalized, origin='lower', extent=(xmin,xmax,ymin,ymax),
                              cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                ax.set_aspect('equal')
                ax.grid(False, which='both')
                ax.set_axis_off()
                ax.text(.5, -.025, titles[col], transform=ax.transAxes,
                        ha='center', va='top', fontsize=12, fontfamily='serif')
        fig.subplots_adjust(hspace=.2, wspace=.06)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.04, facecolor='white')
    finally:
        plt.close(fig)
    return low, high


# *=================== 6. 加载真实数据与权重，提取 BEV ===================
def visualize_real_frames(args):
    import sys
    import json
    from pathlib import Path
    from copy import deepcopy
    import numpy as np
    import yaml

    local_root = Path(__file__).resolve().parents[1]
    # 先定位本地 models 包，再选择含已编译扩展的 mmdet3d 根目录。
    # 这里是普通 import，不再使用 AST、exec 或替换父类。
    sys.path.insert(0, str(local_root))
    import models
    import datasets
    sys.path.insert(0, str(Path(args.daocc_root).expanduser().resolve()))
    from models.TemporalDAOcc.model import TemporalDAOcc
    from datasets.temporal_daocc import TemporalNuScenesDataset
    from torchpack.utils.config import Config as TorchpackConfig
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from tqdm import tqdm

    # 明确加载时序配置的 base_config，与 run.py 的配置继承一致。
    path = Path(args.config).expanduser().resolve()
    overlay = yaml.safe_load(path.read_text())
    config = TorchpackConfig()
    config.load(str((path.parent/overlay.pop('base_config')).resolve()), recursive=True)
    config.update(overlay)
    cfg = Config(recursive_eval(config))
    data_cfg = deepcopy(cfg.data[args.split])

    #! 使用 pair 模式：每次取当前样本，同时独立读取其同场景上一帧。
    # 这里只在选定 split 的 info 中找历史，不跨训练/验证集查找。
    data_cfg.update(test_mode=True, temporal_mode='pair')
    data_cfg.pop('samples_per_gpu', None)
    #! 历史在 datasets/temporal_daocc.py 中按索引偏移选择，跨场景用当前占位。
    dataset = build_dataset(data_cfg)
    if not isinstance(dataset, TemporalNuScenesDataset):
        raise ValueError('请使用 temporal_daocc.yaml 时序配置')

    #! 先确定历史配对，再抽样当前帧；不是把上一次可视化的帧当成历史。
    # 默认展示全部历史；可选参数保留单槽位查看方式，不把多帧混合成一帧。
    if args.history_slot is not None and not 0 <= args.history_slot < len(dataset.history_offsets):
        raise ValueError('--history-slot 超出配置的历史槽位范围')
    slots = list(range(len(dataset.history_offsets))) if args.history_slot is None else [args.history_slot]
    # 原策略只要求一个有效历史，会选到场景开头历史不足的帧：
    # candidates = [i for i in dataset.stream_indices if any(dataset.history_valid[i][s] for s in slots)]
    #! 先要求本次展示的所有历史槽位均有效，再均匀抽样，避免缺失历史造成空白行。
    # 默认检查全部配置历史；指定 --history-slot 时仅检查该槽位。
    candidates = [i for i in dataset.stream_indices if all(dataset.history_valid[i][s] for s in slots)]
    if not candidates:
        raise ValueError('没有具备全部所选历史帧的样本，请减少历史偏移或检查数据集及时间间隔限制')
    count = min(args.num_samples, len(candidates))
    positions = [len(candidates)//2] if count == 1 else np.linspace(
        0, len(candidates)-1, count).round().astype(int).tolist()
    indices = [candidates[i] for i in positions]
    print(f'完整历史筛选：{len(candidates)}/{len(dataset)} 帧符合条件；'
          f'请求 {args.num_samples} 组，实际可视化 {count} 组', flush=True)
    # --num-samples 控制展示多少组；--history-slot 指定每组查看哪个历史偏移。

    torch.cuda.set_device(args.device)
    set_random_seed(0, deterministic=True)
    model_cfg = deepcopy(cfg.model)
    # 父类构造时会调用 init_weights；完整 checkpoint 已包含 backbone 权重。
    # 清除额外的分类预训练初始化，避免重复加载及 fc.weight/fc.bias 警告。
    backbone = model_cfg.encoders.camera.backbone
    if (backbone.get('init_cfg') or {}).get('type') == 'Pretrained':
        backbone.init_cfg = None
    if 'pretrained' in backbone:
        backbone.pretrained = None
    model = build_model(model_cfg)
    if not isinstance(model, TemporalDAOcc):
        raise ValueError('配置中的 model.type 必须为 TemporalDAOcc')
    # 原单帧权重仅允许缺少新增门控参数；其余缺失/多余参数视为配置不匹配。
    # 上面已关闭额外的 backbone 预训练权重加载。
    checkpoint = load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    weights = checkpoint.get('state_dict', checkpoint)
    keys = {k[7:] if k.startswith('module.') else k for k in weights}
    expected = set(model.state_dict())
    missing = sorted(k for k in expected-keys if not k.startswith('temporal_fusion.'))
    unexpected = sorted(keys-expected)
    if missing or unexpected:
        raise ValueError(f'权重与模型不匹配，缺少：{missing}；多余：{unexpected}')
    if args.precision == 'fp16':
        wrap_fp16_model(model)
    model.cuda().eval()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # 场景名优先从 nuScenes 表读取，没有表时使用 scene_token，避免跨场景混存。
    scene_file = Path(dataset.dataset_root)/dataset.version/'scene.json'
    scene_names = {s['token']: s['name'] for s in json.loads(scene_file.read_text())} if scene_file.exists() else {}
    frame_numbers = {}
    for index in dataset.stream_indices:
        scene = dataset.data_infos[index]['scene_token']
        frame_numbers.setdefault(scene, {})[index] = len(frame_numbers.get(scene, {}))+1
    report = dict(config=str(path), checkpoint=str(Path(args.ckpt).expanduser().resolve()),
                  feature_layer='decoder.neck, before temporal fusion', precision=args.precision,
                  bounds=list(model.bev_bounds), reduction='mean(abs(channels))',
                  normalization='shared p01/p99 across all panels for each current frame',
                  coordinates='raw augmented LiDAR BEV, horizontal X / vertical Y; not occupancy ego crop',
                  history_panel='previous-frame coordinates', aligned_panel='current-frame coordinates',
                  invalid_history='white', history_offsets=dataset.history_offsets,
                  history_slots=slots, order='oldest to newest by timestamp; invalid slots last',
                  selection_policy='all selected history slots must be valid',
                  total_samples=len(dataset), eligible_samples=len(candidates),
                  requested_samples=args.num_samples, selected_indices=indices,
                  layout='rows: histories; columns: original, aligned to current, current reference',
                  frame_number='1-based within this dataset scene', frames=[])
    with torch.no_grad():
        for index in tqdm(indices, desc='时序 BEV 对比', unit='组'):
            
            #! dataset[index] → prepare_test_data(index) → _prepare(index, False)。
            # _prepare 按 history_indices[index] 读取各历史槽位，分别运行原 pipeline。
            # collate 组装 batch，scatter 搬到 GPU；这两步不负责查找历史帧。
            batch = collate([dataset[index]], samples_per_gpu=1)
            data = scatter(batch, [args.device])[0]
            history_data = data.pop('history')
            # pop 将历史输入取出，避免 model(**data) 自动执行成对融合逻辑。
            meta = data['temporal_meta'][0]
            histories = history_data if isinstance(history_data, list) else [history_data]
            history_metas = meta['history'] if isinstance(meta['history'], list) else [meta['history']]
            history_valid = meta['history_valid'] if isinstance(meta['history_valid'], list) else [meta['history_valid']]
            #! 当前 BEV 只计算一次，历史逐个计算；仅保留 CPU 热力图，避免积累多帧特征显存。
            current = model(**data, _bev_only=True)
            current_heat = current[0].float().abs().mean(0).cpu().numpy().T
            gcur = torch.as_tensor(meta['lidar2global'], device=current.device, dtype=torch.float32)[None]
            current_inverse_aug = torch.inverse(data['lidar_aug_matrix'].float())
            scene = meta['scene']
            number = frame_numbers[scene][index]
            #! 有效历史按真实时间从早到晚排列，重复引用保留；无效占位放在末尾。
            ordered_slots = sorted(slots, key=lambda s: (
                not history_valid[s], history_metas[s]['timestamp'], -dataset.history_offsets[s]))
            panels, records = [], []
            for slot in ordered_slots:
                offset = dataset.history_offsets[slot]
                previous_index = dataset.history_indices[index][slot]
                previous_meta = history_metas[slot]
                record = dict(slot=slot, offset=offset, valid=bool(history_valid[slot]),
                              history_index=previous_index, history_token=previous_meta['token'],
                              history_timestamp=previous_meta['timestamp'],
                              time_gap_seconds=meta['timestamp']-previous_meta['timestamp'])
                if not history_valid[slot]:
                    # 占位当前帧不伪装成真实历史，也不进行重复推理。
                    panels.append((None, None, None, f"Offset {offset}: unavailable"))
                    records.append(record)
                    continue
                history = model(**histories[slot], _bev_only=True)
                gprev = torch.as_tensor(previous_meta['lidar2global'], device=current.device, dtype=torch.float32)[None]
                transform = (histories[slot]['lidar_aug_matrix'].float() @ torch.inverse(gprev)
                             @ gcur @ current_inverse_aug)
                aligned, valid = warp_bev(history, transform, model.bev_bounds)
                mask = valid[0, 0].cpu().numpy().T
                previous_number = frame_numbers[scene][previous_index]
                label = f"Frame {previous_number}, t-{record['time_gap_seconds']:.2f}s"
                panels.append((history[0].float().abs().mean(0).cpu().numpy().T,
                               aligned[0].float().abs().mean(0).cpu().numpy().T, mask, label))
                record.update(history_frame_number=previous_number,
                              current_to_previous=transform[0].cpu().tolist(),
                              valid_fraction=float(mask.mean()))
                records.append(record)
                del history, aligned, valid
            suffix = 'all_history' if args.history_slot is None else f'offset_{dataset.history_offsets[args.history_slot]}'
            filename = out_dir/scene_names.get(scene, scene)/f"frame_{number:03d}_{suffix}_{meta['token']}.png"
            low, high = save_history_sequence(current_heat, panels, model.bev_bounds, filename, number)
            report['frames'].append(dict(index=index, token=meta['token'], scene=scene,
                scene_name=scene_names.get(scene, scene), frame_number=number,
                timestamp=meta['timestamp'], histories=records,
                shape=list(current.shape), heat_p01=low, heat_p99=high, image=str(filename)))
            manifest = f'temporal_bev_{suffix}_manifest.json'
            (out_dir/manifest).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            del data, history_data, histories, batch, current
    print(f'已保存 {len(report["frames"])} 组历史序列图：{out_dir}')


# *=================== 7. 按下面的顺序执行，无自动发现机制 ===================
def check_ema():
    #* （temporal) 连续三帧递归更新，证明 EMA 保留早于上一帧的信息且不增加参数。
    layer = TemporalFusion(channels=1, mode='ema', alpha=.5)
    assert sum(p.numel() for p in layer.parameters()) == 0
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    memory = torch.zeros(1, 1, 2, 2)  # 首帧 F1=0。
    memory = layer(torch.full_like(memory, 2.), memory, valid)  # M2=1。
    memory = layer(torch.full_like(memory, 4.), memory, valid)  # M3=2.5，而非仅两帧平均3。
    torch.testing.assert_close(memory, torch.full_like(memory, 2.5))
    reset = layer(torch.full_like(memory, 6.), memory, torch.zeros_like(valid))
    assert torch.equal(reset, torch.full_like(memory, 6.))
    print('  通过：EMA 零参数、连续更新与无效历史重置')


def run_cpu_checks():
    print("1/4 检查 BEV 对齐")
    check_bev_alignment()

    print("2/4 检查特征融合")
    check_feature_fusion()

    print("3/4 检查梯度")
    check_fusion_gradients()

    print("4/4 检查位姿与增强")
    check_pose_alignment()

    print('额外检查 EMA 递归更新')
    check_ema()

    # 任一 assert 失败，程序会在对应位置报错，不会执行到这里。
    print("全部通过。以上仅为 CPU 对齐/融合检查，不包含真实数据配对和完整模型缓存测试。")


def main():
    import argparse
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='无参数执行 CPU 检查；--visualize 保存真实帧的 BEV 对比图')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--config', default=str(root/'configs/temporal_daocc/temporal_daocc.yaml'))
    parser.add_argument('--ckpt', help='原始 DAOcc 或训练后的 TemporalDAOcc PyTorch 权重')
    parser.add_argument('--daocc-root', default=str(root))
    parser.add_argument('--out-dir')
    parser.add_argument('--num-samples', type=int, default=10, help='均匀取 N 对有效相邻帧；不足则全部取出')
    parser.add_argument('--history-slot', type=int, default=None, help='默认展示全部历史；指定后仅查看一个槽位，从0开始')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()
    if args.visualize:
        if not args.ckpt or not args.out_dir or args.num_samples < 1:
            parser.error('可视化需指定 --ckpt、--out-dir，且 --num-samples > 0')
        visualize_real_frames(args)
    else:
        run_cpu_checks()


if __name__ == '__main__':
    main()
