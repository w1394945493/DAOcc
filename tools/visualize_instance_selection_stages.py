#!/usr/bin/env python3
"""可视化实例框从原始检测候选到训练/推理最终选择的各阶段。

同一组 eval-mode 预测用于比较筛选规则，避免网络前向差异干扰选择结果：
原始 Top-K、GT、初步筛选、训练 score_only、训练 gt_quality、推理实例、检测指标框。
训练策略面板是对同一预测结果的离线规则模拟，不执行反向传播或修改模型。
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / 'configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_instance_bevfusion.yaml'
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='模型权重；不指定时使用随机初始化模型')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs/instance_selection_stages')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='均匀采样的起始索引')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='从起始索引到验证集末尾均匀采样的数量')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--raw-display-limit', type=int, default=150)
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--no-train-score-filter', action='store_true',
                        help='兼容旧命令；现在始终同时展示仅几何、仅得分、得分+几何三种策略')
    parser.add_argument('--box-cost-threshold', type=float, default=1.5)
    parser.add_argument('--iou-threshold', type=float, default=0.3)
    parser.add_argument('--class-aware', dest='class_aware', action='store_true', default=True)
    parser.add_argument('--no-class-aware', dest='class_aware', action='store_false')
    args = parser.parse_args()
    if min(args.num_samples, args.raw_display_limit) < 1:
        parser.error('num-samples和raw-display-limit必须为正')
    if args.sample_index < 0 or not 0 <= args.score_threshold <= 1:
        parser.error('sample-index须非负，score-threshold须在[0,1]内')
    if args.box_cost_threshold < 0 or not 0 <= args.iou_threshold <= 1:
        parser.error('box-cost-threshold须非负，iou-threshold须在[0,1]内')
    return args


def training_initial_ids(head, raw, meta):
    """范围/检测阈值/NMS/实例阈值/合法性检查之后的原始候选编号。"""
    import torch
    selected = []
    for task in range(len(head.num_classes)):
        ids = (raw['task'] == task).nonzero(as_tuple=True)[0]
        local = head._select_task(raw['boxes'][ids], raw['scores'][ids],
                                  raw['local_labels'][ids], task, meta,
                                  for_instances=True)
        selected.append(ids[local])
    return torch.cat(selected) if selected else raw['scores'].new_empty(0, dtype=torch.long)


def inference_initial_ids(head, raw, meta, normal_selection):
    """检测最终框中满足实例合法性和实例分数阈值的原始候选编号。"""
    ids = normal_selection(head, raw, meta['box_type_3d'])
    boxes, scores = raw['boxes'][ids], raw['scores'][ids]
    valid = boxes.isfinite().all(-1) & scores.isfinite() & (boxes[:, 3:6] > 0).all(-1)
    valid &= scores >= head.instance_score_threshold
    return ids[valid], ids


def gt_quality_ids(head, raw, ids, meta, gt_boxes, gt_labels):
    """调用生产代码的GT质量筛选，并用candidate_ids保留原始候选编号。"""
    item = dict(boxes=raw['boxes'][ids], scores=raw['scores'][ids],
                labels=raw['labels'][ids], source_indices=raw['source'][ids],
                task_ids=raw['task'][ids], candidate_ids=ids)
    filtered = head._filter_instances_by_gt(item, meta, gt_boxes, gt_labels)
    return filtered['candidate_ids']


def occ_final_ids(fusion, sampler, raw, ids, query_points):
    """应用OCC空间交集和max_instances，返回原始候选编号。"""
    import torch
    if not len(ids):
        return ids
    ids = ids[raw['scores'][ids].argsort(descending=True)]
    item = dict(boxes=raw['boxes'][ids], scores=raw['scores'][ids],
                labels=raw['labels'][ids])
    bounds = (sampler.lidar_x_min, sampler.lidar_y_min,
              sampler.lidar_x_max, sampler.lidar_y_max)
    src, _, _ = fusion.build_edges(query_points, item, bounds,
                                    (sampler.output_h, sampler.output_w))
    if not src.numel():
        return ids[:0]
    # item已按分数降序；build_edges也按该顺序执行max_instances筛选。
    used = torch.unique(src, sorted=True)
    return ids[used]


def associated_grid_ids(fusion, sampler, raw, ids, query_points):
    """返回筛选实例框实际覆盖的唯一 OCC BEV 网格编号。"""
    item = dict(boxes=raw['boxes'][ids], scores=raw['scores'][ids],
                labels=raw['labels'][ids])
    bounds = (sampler.lidar_x_min, sampler.lidar_y_min,
              sampler.lidar_x_max, sampler.lidar_y_max)
    _, dst, _ = fusion.build_edges(query_points, item, bounds,
                                   (sampler.output_h, sampler.output_w))
    #! dst 是实际参与“实例框 -> OCC 网格”连边的目标网格编号；去重后用于可视化。
    return dst.unique(sorted=True)


def draw_box(ax, box, color, linewidth=1.0, alpha=1.0):
    import numpy as np
    from matplotlib.patches import Polygon
    local = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) * box[3:5] / 2
    c, s = np.cos(box[6]), np.sin(box[6])
    corners = local @ np.array([[c, s], [-s, c]]) + box[:2]
    ax.add_patch(Polygon(corners, fill=False, edgecolor=color,
                         linewidth=linewidth, alpha=alpha))
    front = (corners[2] + corners[3]) / 2
    ax.plot([box[0], front[0]], [box[1], front[1]], color=color,
            linewidth=linewidth, alpha=alpha)


def draw_panel(ax, title, boxes, points, color, shown=None):
    import numpy as np
    finite = np.isfinite(points[:, :2]).all(-1)
    ax.scatter(points[finite, 0], points[finite, 1], s=.12, c='.45', alpha=.35,
               linewidths=0, rasterized=True)
    display = boxes if shown is None else boxes[:shown]
    for box in display:
        draw_box(ax, box, color)
    suffix = '' if shown is None or len(boxes) <= shown else f'; shown {shown}'
    ax.set_title(f'{title}: {len(boxes)}{suffix}', fontsize=10)
    ax.set_xlim(-54, 54); ax.set_ylim(-54, 54)
    ax.set_aspect('equal'); ax.grid(alpha=.12)
    ax.set_xlabel('LiDAR X (m)'); ax.set_ylabel('LiDAR Y (m)')


def draw_grid_panel(ax, title, boxes, grid_ids, occ_labels, palette, color, grid_corners):
    """以 OCC GT 为底图，突出实例框关联的非空语义网格。"""
    import numpy as np
    from matplotlib.collections import PolyCollection, LineCollection
    height, width = occ_labels.shape
    occupied = (occ_labels >= 0) & (occ_labels < 17)
    associated = np.zeros(height * width, dtype=bool)
    associated[grid_ids.astype(np.int64)] = True
    associated = associated.reshape(height, width) & occupied

    #! 空体素（free=17、ignore=255）不填色；其他非空 OCC GT 网格显示为灰色。
    ys, xs = np.nonzero(occupied)
    polygons = np.stack([grid_corners[ys, xs], grid_corners[ys, xs + 1],
                         grid_corners[ys + 1, xs + 1], grid_corners[ys + 1, xs]], axis=1)
    facecolors = np.full((len(ys), 4), [0.58, 0.58, 0.58, 0.72])
    selected = associated[ys, xs]
    #! 框内且 GT 非空的网格，按该 XY 柱最高非空体素的 OCC 语义类别着色。
    facecolors[selected, :3] = palette[occ_labels[ys[selected], xs[selected]].astype(int)] / 255.
    facecolors[selected, 3] = .95
    ax.add_collection(PolyCollection(polygons, facecolors=facecolors,
                                    edgecolors='none', zorder=1))
    #! 保留整个 OCC 范围的真实网格线，包括空单元；空体素不填色不等于隐藏网格。
    lines = [grid_corners[:, j] for j in range(width + 1)]
    lines += [grid_corners[i, :] for i in range(height + 1)]
    ax.add_collection(LineCollection(lines, colors='.48', linewidths=.30,
                                     alpha=.65, zorder=2))
    for box in boxes:
        draw_box(ax, box, color, linewidth=.35)
    ax.set_title(f'{title}: {len(boxes)} boxes, {associated.sum()} occupied grids', fontsize=10)
    ax.set_xlim(-54, 54); ax.set_ylim(-54, 54)
    ax.set_aspect('equal'); ax.grid(alpha=.12)
    ax.set_xlabel('LiDAR X (m)'); ax.set_ylabel('LiDAR Y (m)')


def draw_overview(path, stages, grids, occ_labels, palette, points,
                  grid_corners, raw_limit, note):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(22, 17))
    panels = [
        ('Raw prediction Top-K', stages['raw'], 'tab:gray', raw_limit),
        ('GT boxes', stages['gt'], 'black', None),
        ('Initial instance filtering', stages['initial'], 'tab:blue', None),
        ('Final 3D detections / metric', stages['detections'], 'tab:red', None),
        #! 三种训练策略与第四列推理结果，和第三行OCC网格逐列对应。
        ('Train final: geometry only', stages['train_geometry_only'], 'tab:cyan', None),
        ('Train final: score_only', stages['train_score_only'], 'tab:green', None),
        ('Train final: score + geometry', stages['train_gt_quality'], 'tab:orange', None),
        #! 与第三行第四列的推理 OCC GT 关联图上下对应。
        ('Inference final instances', stages['inference'], 'tab:purple', None),
    ]
    for ax, (title, boxes, color, shown) in zip(axes.flat, panels):
        draw_panel(ax, title, boxes, points, color, shown)
    #! 四个子图分别展示三种训练策略及推理策略最终实例实际关联的 OCC 网格。
    grid_panels = [
        ('Train grids: geometry only', 'train_geometry_only', 'tab:cyan'),
        ('Train grids: score_only', 'train_score_only', 'tab:green'),
        ('Train grids: score + geometry', 'train_gt_quality', 'tab:orange'),
        ('Inference grids', 'inference', 'tab:purple'),
    ]
    for ax, (title, name, color) in zip(axes[2], grid_panels):
        draw_grid_panel(ax, title, stages[name], grids[name], occ_labels,
                        palette, color, grid_corners)
    fig.text(.02, .01, note.replace('\n', '  '), fontsize=8)
    fig.suptitle('Instance-box selection stages (same eval-mode predictions)', fontsize=13)
    fig.tight_layout(rect=[0, .03, 1, .97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
    from torch.utils.data import Subset
    from torchpack.utils.config import Config as TorchpackConfig
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import recursive_eval
    from visualize_instance_bev_relations import decode_candidates, normal_selection
    from visualize_modality_comparison import (clear_pretrained, first_tensor,
                                                load_occ_colors, top_labels)

    set_random_seed(0, deterministic=True)
    raw_cfg = TorchpackConfig(); raw_cfg.load(str(args.config.resolve()), recursive=True)
    cfg = Config(recursive_eval(raw_cfg))
    if cfg.model.heads.object.type != 'ProposalHead' or cfg.model.heads.occ.type != 'InstanceBEVOCCHead2D':
        raise ValueError('脚本要求ProposalHead + InstanceBEVOCCHead2D配置')

    data_cfg = deepcopy(cfg.data.val)
    data_cfg.test_mode = True
    data_cfg.pop('samples_per_gpu', None)
    dataset = build_dataset(data_cfg)
    if args.sample_index >= len(dataset):
        raise ValueError(f'起始索引{args.sample_index}超出验证集长度{len(dataset)}')
    available = len(dataset) - args.sample_index
    if args.num_samples > available:
        raise ValueError(f'num-samples={args.num_samples}超过起始索引后的样本数{available}')
    #! 从 sample-index 到验证集最后一帧（包含两端）均匀选取 num-samples 帧。
    if args.num_samples == 1:
        indices = [args.sample_index]
    else:
        end = len(dataset) - 1
        indices = [round(args.sample_index + i * (end - args.sample_index)
                         / (args.num_samples - 1)) for i in range(args.num_samples)]
    loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                              workers_per_gpu=0, dist=False, shuffle=False)

    torch.cuda.set_device(args.gpu)
    model_cfg = deepcopy(cfg.model); clear_pretrained(model_cfg)
    model = build_model(model_cfg)
    # 2026-11-23：允许跳过权重，便于只检查数据、筛选及可视化流程。
    #! 不加载权重时预测框来自随机初始化模型，不代表真实检测效果。
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = str(args.checkpoint.resolve())
        load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    model.float().eval().cuda(args.gpu)
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False

    head = model.heads['object']
    occ_head = model.heads['occ']
    fusion = occ_head.instance_fusion
    sampler = occ_head.coordinate_transform
    palette = load_occ_colors(ROOT / 'tools/vis_occ3d.py')
    # 可视化阈值独立于配置，默认值与当前实验一致。
    head.instance_score_threshold = args.score_threshold
    if args.no_train_score_filter:
        head.instance_train_score_filter = False
    head.instance_box_cost_threshold = args.box_cost_threshold
    head.instance_iou_threshold = args.iou_threshold
    head.instance_class_aware = args.class_aware

    args.output_dir.mkdir(parents=True, exist_ok=True)
    def cpu(value):
        return value.detach().cpu().numpy()

    with torch.no_grad():
        for dataset_index, batch in zip(indices, loader):
            data = scatter(batch, [args.gpu])[0]
            task_outputs = {}
            handles = []
            for task_id, task_head in enumerate(head.task_heads):
                def capture(_module, _inputs, output, task_id=task_id):
                    task_outputs[task_id] = output
                handles.append(task_head.register_forward_hook(capture))
            try:
                outputs = model(**data)
            finally:
                for handle in handles:
                    handle.remove()
            predictions = tuple([task_outputs[i]] for i in range(len(head.task_heads)))
            raw = decode_candidates(head, predictions)
            meta = data['metas'][0]
            #! 前向仍为eval；仅离线模拟训练筛选时临时切换head标志，随后立即恢复。
            was_training = head.training
            was_score_filter = head.instance_train_score_filter
            try:
                head.training = True
                #! 同一组候选独立模拟三种策略，不让配置开关改变“仅得分”面板含义。
                head.instance_train_score_filter = False
                geometry_initial = training_initial_ids(head, raw, meta)
                initial = geometry_initial[raw['scores'][geometry_initial] >= head.instance_score_threshold]
            finally:
                head.training = was_training
                head.instance_train_score_filter = was_score_filter
            inference_initial, detection_ids = inference_initial_ids(
                head, raw, meta, normal_selection)

            gt_boxes = first_tensor(data['gt_bboxes_3d'])
            gt_labels = first_tensor(data['gt_labels_3d']).long()
            quality = gt_quality_ids(head, raw, initial, meta, gt_boxes, gt_labels)
            geometry = gt_quality_ids(head, raw, geometry_initial, meta, gt_boxes, gt_labels)
            matrices = [first_tensor(data[key]) for key in
                        ('lidar_aug_matrix', 'lidar2ego', 'occ_aug_matrix')]
            query_points = fusion.query_points(sampler, *matrices)[0]
            train_score = occ_final_ids(fusion, sampler, raw, initial, query_points)
            train_quality = occ_final_ids(fusion, sampler, raw, quality, query_points)
            train_geometry = occ_final_ids(fusion, sampler, raw, geometry, query_points)
            inference = occ_final_ids(fusion, sampler, raw, inference_initial, query_points)
            grids_torch = dict(
                train_geometry_only=associated_grid_ids(
                    fusion, sampler, raw, train_geometry, query_points),
                train_score_only=associated_grid_ids(
                    fusion, sampler, raw, train_score, query_points),
                train_gt_quality=associated_grid_ids(
                    fusion, sampler, raw, train_quality, query_points),
                inference=associated_grid_ids(
                    fusion, sampler, raw, inference, query_points))

            # 检测面板直接使用模型最终输出；同时核对离线检测筛选没有偏离指标输入。
            expected = outputs[0]
            detection_boxes = expected['boxes_3d'].tensor.clone()
            detection_boxes[:, :3] = expected['boxes_3d'].gravity_center
            torch.testing.assert_close(raw['boxes'][detection_ids, :7].cpu(),
                                       detection_boxes[:, :7].cpu(), rtol=1e-4, atol=1e-4)

            gt_center = gt_boxes.tensor.clone(); gt_center[:, :3] = gt_boxes.gravity_center
            raw_order = raw['scores'].argsort(descending=True)
            stages_torch = dict(
                raw=raw['boxes'][raw_order], gt=gt_center,
                train_geometry_only=raw['boxes'][train_geometry],
                initial=raw['boxes'][initial], train_score_only=raw['boxes'][train_score],
                train_gt_quality=raw['boxes'][train_quality],
                inference=raw['boxes'][inference], detections=detection_boxes)
            stages = {name: cpu(value) for name, value in stages_torch.items()}
            grids = {name: cpu(value) for name, value in grids_torch.items()}
            query_grid = cpu(query_points).reshape(sampler.output_h, sampler.output_w, 3)
            #! 查询网格可能经过旋转增强；用两个实际步进向量构造全部网格角点。
            row_step = np.diff(query_grid[..., :2], axis=0).mean(axis=(0, 1))
            col_step = np.diff(query_grid[..., :2], axis=1).mean(axis=(0, 1))
            origin = query_grid[0, 0, :2] - row_step / 2 - col_step / 2
            row_ids = np.arange(sampler.output_h + 1)[:, None, None]
            col_ids = np.arange(sampler.output_w + 1)[None, :, None]
            grid_corners = origin + row_ids * row_step + col_ids * col_step
            #! top_labels 仅消去 Z，返回 [X,Y]；必须转置为与 query_points/dst 一致的 [Y,X]。
            #! 方形网格无法通过 shape 检查发现漏转置；漏转置会把 GT 语义贴到错误位置。
            semantics = first_tensor(data['voxel_semantics'])
            if semantics.ndim == 4:
                semantics = semantics[0]
            occ_labels = top_labels(cpu(semantics)).T
            if occ_labels.shape != (sampler.output_h, sampler.output_w):
                raise ValueError(f'OCC GT BEV形状{occ_labels.shape}与查询网格不一致')
            lidar_points = cpu(first_tensor(data['points']))[:, :3]
            token = dataset.data_infos[dataset_index]['token']
            folder = args.output_dir / f'{dataset_index:06d}_{token}'
            folder.mkdir(parents=True, exist_ok=True)
            note = ('Initial = range + detector threshold + NMS + instance score + valid geometry.\n'
                    'Final instance sets additionally apply OCC overlap and max_instances.\n'
                    'gt_quality = same-class (box_cost < threshold OR 3D IoU > threshold).')
            note += '\nGeometry only skips the instance score threshold; detector thresholds remain.'
            draw_overview(folder / 'selection_stages.png', stages, grids, occ_labels, palette,
                          lidar_points, grid_corners, args.raw_display_limit, note)

            frame = dict(index=dataset_index, token=token,
                         counts={name: len(value) for name, value in stages.items()},
                         output=str((folder / 'selection_stages.png').resolve()))
            print(json.dumps(frame, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
