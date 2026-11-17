#!/usr/bin/env python3
"""独立分析 CenterHead 实例、预测框、占据 BEV 和 GT 的空间关联。

仅新增此脚本：使用只读 forward hooks，不替换模型方法、不修改预测或权重。
输出 overview.png、overview_ego.png、relations.npz 和 manifest.json。
高置信度仅指通过筛选，并不保证实际定位质量；热图为通道 RMS，不是注意力。
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / 'configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_iter.yaml'
DEFAULT_CHECKPOINT = '/home/wangyushen/Data/Outputs/daocc/train/daocc/w_mask/train_spconv2_voxel_iter_fp16_htc/latest.pth'


# Occ3D 中对应检测任务的全部10类前景，包含静态的屏障和交通锥；其余占据类别灰色。
FOREGROUND_CLASSES = {1: 'barrier', 2: 'bicycle', 3: 'bus', 4: 'car', 5: 'construction vehicle',
                   6: 'motorcycle', 7: 'pedestrian', 8: 'traffic cone', 9: 'trailer', 10: 'truck'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/instance_bev_relations')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--sample-index', type=int, default=0, help='验证集起始帧索引（从0开始）')
    p.add_argument('--num-samples', type=int, default=1, help='采样帧数')
    p.add_argument('--sample-stride', type=int, default=1,
                   help='验证集索引步长：start + i * stride；默认1连续采样，不保证跨场景')
    p.add_argument('--score-threshold', type=float, default=0.3,
                   help='关联框阈值，实验起点而非经过标定的质量保证')
    p.add_argument('--max-instances', type=int, default=100)
    p.add_argument('--display-candidates', type=int, default=150)
    p.add_argument('--camera-mask', action='store_true')
    args = p.parse_args()
    if not 0 <= args.score_threshold <= 1 or args.sample_index < 0:
        p.error('分数应在[0,1]，sample-index应非负')
    if min(args.num_samples, args.sample_stride, args.max_instances, args.display_candidates) < 1:
        p.error('样本数/采样步长/实例上限/显示上限应为正')
    return args


def decode_candidates(head, predictions):
    """沿用原 coder 的坐标/回归约定，但在过滤前保存 Top-K 来源索引。"""
    import torch
    records = []
    offset = 0
    coder = head.bbox_coder
    for task_id, task_predictions in enumerate(predictions):
        pred = task_predictions[0]
        scores, inds, classes, ys, xs = coder._topk(pred['heatmap'].float().sigmoid(), K=coder.max_num)
        def gather(key):
            return coder._transpose_and_gather_feat(pred[key].float(), inds)
        reg = gather('reg')
        xs = (xs[..., None] + reg[..., :1]) * coder.out_size_factor * coder.voxel_size[0] + coder.pc_range[0]
        ys = (ys[..., None] + reg[..., 1:2]) * coder.out_size_factor * coder.voxel_size[1] + coder.pc_range[1]
        dim = gather('dim')
        if head.norm_bbox:
            dim = dim.exp()
        rot = gather('rot')
        yaw = torch.atan2(rot[..., :1], rot[..., 1:2])
        parts = [xs, ys, gather('height'), dim, yaw]
        if 'vel' in pred:
            parts.append(gather('vel'))
        boxes = torch.cat(parts, -1)[0]
        records.append(dict(boxes=boxes, scores=scores[0], labels=classes[0].long() + offset,
                            local_labels=classes[0].long(), source=inds[0],
                            task=torch.full_like(inds[0], task_id)))
        offset += head.num_classes[task_id]
    return {key: torch.cat([r[key] for r in records]) for key in records[0]}


def normal_selection(head, raw, box_type):
    """使用原 NMS 算子和配置，显式传递候选行号；结果必须与原模型逐项核对。"""
    import torch
    from mmdet3d.core import circle_nms, xywhr2xyxyr
    from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
    cfg, coder = head.test_cfg, head.bbox_coder
    selected = []
    for task in range(len(head.num_classes)):
        ids = (raw['task'] == task).nonzero(as_tuple=True)[0]
        boxes, scores = raw['boxes'][ids], raw['scores'][ids]
        lower = boxes.new_tensor(coder.post_center_range[:3])
        upper = boxes.new_tensor(coder.post_center_range[3:])
        mask = ((boxes[:, :3] >= lower) & (boxes[:, :3] <= upper)).all(-1)
        if coder.score_threshold:
            mask &= scores > coder.score_threshold
        ids = ids[mask]
        nms_type = cfg['nms_type'][task] if isinstance(cfg['nms_type'], list) else cfg['nms_type']
        if nms_type == 'circle':
            xy_score = torch.cat([raw['boxes'][ids, :2], raw['scores'][ids, None]], -1)
            keep = circle_nms(xy_score.cpu().numpy(), cfg['min_radius'][task], post_max_size=cfg['post_max_size'])
            ids = ids[torch.as_tensor(keep, device=ids.device, dtype=torch.long)]
        else:
            ids = ids[raw['scores'][ids] >= cfg['score_threshold']]
            if len(ids):
                bev = box_type(raw['boxes'][ids], coder.code_size).bev
                scales = cfg.get('nms_scale', 1.0)
                scales = scales[task] if isinstance(scales, list) else [scales] * head.num_classes[task]
                for label, scale in enumerate(scales):
                    mask = raw['local_labels'][ids] == label
                    bev[mask, 2:4] *= scale
                keep = nms_gpu(xywhr2xyxyr(bev), raw['scores'][ids], cfg['nms_thr'],
                               pre_maxsize=cfg['pre_max_size'], post_max_size=cfg['post_max_size'])
                ids = ids[keep]
                bounds = raw['boxes'].new_tensor(cfg['post_center_limit_range'])
                centers = raw['boxes'][ids, :3]
                ids = ids[((centers >= bounds[:3]) & (centers <= bounds[3:])).all(-1)]
        selected.append(ids)
    return torch.cat(selected)


def occ_to_lidar(points, lidar_aug, lidar2ego, occ_aug):
    """与 CrossCoordinateSample 的 ego 分支一致，保留完整 xyz（行向量）。"""
    import numpy as np
    ego = points @ np.linalg.inv(occ_aug[:3, :3]).T
    lidar = (ego - lidar2ego[:3, 3]) @ np.linalg.inv(lidar2ego[:3, :3]).T
    return lidar @ lidar_aug[:3, :3].T + lidar_aug[:3, 3]


def inside_box(points, box, use_height=False):
    import numpy as np
    delta = points - box[:3]
    c, s = np.cos(box[6]), np.sin(box[6])
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    mask = (np.abs(local_x) <= box[3] / 2) & (np.abs(local_y) <= box[4] / 2)
    if use_height:
        mask &= np.abs(delta[:, 2]) <= box[5] / 2
    return mask


def make_edges(points, boxes, use_height=False, valid=None):
    import numpy as np
    edges = []
    for i, box in enumerate(boxes):
        mask = inside_box(points, box, use_height)
        if valid is not None:
            mask &= valid
        ids = np.flatnonzero(mask)
        edges.append(np.stack([np.full(len(ids), i, dtype=np.int64), ids]))
    return np.concatenate(edges, axis=1) if edges else np.empty((2, 0), dtype=np.int64)


def association_rgb(sem, edges, palette, valid, foreground_only=False):
    """[X,Y,Z] GT：关联体素优先，关联柱取最高关联体素，否则灰色占据背景。"""
    import numpy as np
    marked = np.zeros(sem.size, dtype=bool)
    marked[edges[1]] = True
    marked = marked.reshape(sem.shape) & valid
    if foreground_only:
        marked &= np.isin(sem, list(FOREGROUND_CLASSES))
    rgb = np.full((*sem.shape[:2], 3), 255, dtype=np.uint8)
    rgb[valid.any(-1)] = 165
    has = marked.any(-1)
    z = sem.shape[-1] - 1 - marked[..., ::-1].argmax(-1)
    labels = np.take_along_axis(sem, z[..., None], axis=-1)[..., 0]
    rgb[has] = palette[labels[has].astype(int)]
    return rgb.transpose(1, 0, 2)


def foreground_rgb(sem, palette, valid):
    """Overview 的预测/GT 投影：全图检测前景类别优先，每柱取最高前景体素，背景灰色。

    不依赖预测框/关联边；free、ignore 及有效掩码外区域留白。
    多种前景在同一 XY 柱重叠时，俯视图仅显示其中最高者。
    """
    import numpy as np
    foreground = valid & np.isin(sem, list(FOREGROUND_CLASSES))
    rgb = np.full((*sem.shape[:2], 3), 255, dtype=np.uint8)
    rgb[valid.any(-1)] = 165
    has = foreground.any(-1)
    z = sem.shape[-1] - 1 - foreground[..., ::-1].argmax(-1)
    labels = np.take_along_axis(sem, z[..., None], axis=-1)[..., 0]
    rgb[has] = palette[labels[has].astype(int)]
    return rgb.transpose(1, 0, 2)


def heat_rgb(heat, mask):
    import numpy as np
    import matplotlib.pyplot as plt
    low, high = np.percentile(heat, [1, 99])
    normalized = np.clip((heat - low) / max(high - low, 1e-12), 0, 1)
    rgb = plt.get_cmap('gray')(normalized)[..., :3]
    rgb[mask] = plt.get_cmap('turbo')(normalized[mask])[..., :3]
    return rgb


def detection_heat_in_ego(feature, occ_points, shape, source_points):
    """在 ego 查询平面上重采样完整检测特征，再计算 RMS；不旋转已上色图片。

    feature 为 [C,X,Y]，occ_points 为按 [Y,X] 排列的 ego 网格在 LiDAR 系中的位置。
    按检测整数锚点定义归一化坐标，使来源网格索引与插值中心严格对应。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    feature = torch.as_tensor(feature).detach().float().cpu()
    nx, ny = feature.shape[1:]
    dx = source_points[ny, 0] - source_points[0, 0]
    dy = source_points[1, 1] - source_points[0, 1]
    xy = (occ_points[:, :2] - source_points[0, :2]) / np.array([dx, dy])
    normalized = 2 * (xy + .5) / np.array([nx, ny]) - 1
    grid = torch.as_tensor(normalized.reshape(1, *shape, 2), dtype=torch.float32)
    sampled = F.grid_sample(feature.transpose(1, 2)[None], grid,
                            mode='bilinear', padding_mode='zeros', align_corners=False)
    return sampled[0].square().mean(0).sqrt().numpy()


def draw_result(path, raw, normal, selected, det_heat, occ_heat, source_xy, source_points,
                occ_points, gt_rgb, transforms, candidate_limit, title, instance_id=None, foreground_palette=None, gt_boxes=None, gt_association_rgb=None, ego_view=False, ego_det_heat=None, lidar_points=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Patch
    from matplotlib.lines import Line2D
    la, le, oa = transforms
    def to_occ(points):
        lidar = (points - la[:3, 3]) @ np.linalg.inv(la[:3, :3]).T
        ego = lidar @ le[:3, :3].T + le[:3, 3]
        return ego @ oa[:3, :3].T
    def draw_boxes(ax, boxes, ego=False, color='lime'):
        # 预测框绿色，GT 框橙色；颜色不代表占据语义类别。
        for i, box in enumerate(boxes):
            local = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) * box[3:5] / 2
            c, s = np.cos(box[6]), np.sin(box[6])
            xy = local @ np.array([[c,s],[-s,c]]) + box[:2]
            xyz = np.column_stack([xy, np.full(4, box[2])])
            if ego:
                xyz = to_occ(xyz)
            ax.add_patch(Polygon(xyz[:, :2], fill=False, edgecolor=color, linewidth=.65))
            if len(boxes) <= 20:
                ax.text(*xyz[0, :2], str(i if instance_id is None else instance_id), fontsize=6, color='red', clip_on=True)
    # 第二行第一幅：框内彩色仅突出检测特征 RMS 的空间范围，不表示对框内特征做 ROI 池化。
    # 第二行第二幅：彩色表示框在占据 BEV 上的关联范围；此处仅绘图，不执行实例特征注入。
    # 绘图掩码按框的 XY 范围生成；实际 occ_edges 还会剔除超出检测特征有效范围的查询点。
    det_mask = np.zeros(len(source_points), dtype=bool)
    occ_mask = np.zeros(len(occ_points), dtype=bool)
    for box in selected:
        det_mask |= inside_box(source_points, box)
        occ_mask |= inside_box(occ_points, box)
    # source_points/检测特征按[X,Y]展平；显示时转置为[Y,X]。
    det_rgb = heat_rgb(det_heat, det_mask.reshape(det_heat.T.shape).T)
    occ_rgb = heat_rgb(occ_heat, occ_mask.reshape(occ_heat.shape))
    if ego_view:
        if ego_det_heat is None or ego_det_heat.shape != occ_heat.shape:
            raise ValueError('ego 总览需要与占据网格对齐的检测特征热图')
        det_rgb = heat_rgb(ego_det_heat, occ_mask.reshape(occ_heat.shape))
    overview = gt_boxes is not None
    fig, axes = plt.subplots(2, 4 if overview else 3, figsize=(24 if overview else 18, 12))
    titles = [f'Raw Top-K: {len(raw)} (shown {min(len(raw), candidate_limit)})',
              f'Original filtering + NMS: {len(normal)}', f'Association candidates: {len(selected)}',
              'Detection feature RMS / source points (LiDAR)',
              'Occupancy BEV feature RMS (ego)', 'GT: associated occupied voxels in class colors (ego)']
    if overview:
        titles[3:3] = [f'All dataset GT boxes: {len(gt_boxes)} (LiDAR)']
        titles[-1] = 'Selected predictions + all predicted foreground (ego)'
        titles.append('All GT boxes + all GT foreground (ego)')
    if ego_view:
        titles = [name.replace('(LiDAR)', '(ego)') for name in titles]
        titles[:3] = [name + ' (ego)' for name in titles[:3]]
    for ax, name in zip(axes.flat, titles):
        ax.set_title(name, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    # 使用 pipeline 输出的点云，与检测框同处增强后的 LiDAR 系；ego 图应用相同逆增强/外参变换。
    # 只过滤非有限坐标，不以预测框、占据类别或 camera mask 筛选点云。
    if lidar_points is not None:
        points = np.asarray(lidar_points)[:, :3]
        points = points[np.isfinite(points).all(-1)]
        if ego_view:
            points = to_occ(points)
        for ax in axes[0]:
            ax.scatter(points[:, 0], points[:, 1], s=.15, c='.35', alpha=.45,
                       linewidths=0, rasterized=True, zorder=0)
    for ax, boxes in zip(axes[0], [raw[:candidate_limit], normal, selected]):
        ax.set_facecolor('.9'); ax.set_xlim(-54,54); ax.set_ylim(-54,54)
        draw_boxes(ax, boxes, ego_view)
    if overview:
        axes[0,3].set_facecolor('.9'); axes[0,3].set_xlim(-54,54); axes[0,3].set_ylim(-54,54)
        draw_boxes(axes[0,3], gt_boxes, ego_view, color='darkorange')
        axes[1,3].imshow(gt_association_rgb, origin='lower', extent=[-40,40,-40,40])
        draw_boxes(axes[1,3], gt_boxes, True, color='darkorange')
    # 检测整数锚点从 -54 开始；像素边界扩展半格，使来源标记与像素中心重合。
    dx = source_points[det_heat.shape[0], 0] - source_points[0, 0]
    dy = source_points[1, 1] - source_points[0, 1]
    det_extent = [-40,40,-40,40] if ego_view else [-54-dx/2,54-dx/2,-54-dy/2,54-dy/2]
    axes[1,0].imshow(det_rgb, origin='lower', extent=det_extent)
    draw_boxes(axes[1,0], selected, ego_view)
    # 紫色 × 标记产生该预测框的候选来源网格；回归偏移可使预测框中心偏离该点。
    # ego 视图只变换标记的显示坐标，实例向量仍取自原始检测 BEV 的同一来源索引。
    if len(source_xy):
        if ego_view:
            source_xy = to_occ(np.column_stack([source_xy, np.zeros(len(source_xy))]))[:, :2]
        axes[1,0].scatter(source_xy[:,0], source_xy[:,1], s=14, c='magenta', marker='x')
    axes[1,1].imshow(occ_rgb, origin='lower', extent=[-40,40,-40,40])
    axes[1,2].imshow(gt_rgb, origin='lower', extent=[-40,40,-40,40])
    # 框投影使用完整坐标变换；中心高度的轮廓仅供参考，关联以各网格实际映射点为准。
    draw_boxes(axes[1,1], selected, True)
    draw_boxes(axes[1,2], selected, True)
    if instance_id is not None and len(selected) == 1:
        center = selected[0, :3]
        radius = max(5., float(max(selected[0, 3:5])) * 1.5)
        ego_center = to_occ(center[None])[0]
        for ax, point in [(axes[0,2], center), (axes[1,0], center),
                          (axes[1,1], ego_center), (axes[1,2], ego_center)]:
            ax.set_xlim(point[0]-radius, point[0]+radius)
            ax.set_ylim(point[1]-radius, point[1]+radius)
    if ego_view:
        for ax in axes.flat:
            ax.set_xlim(-40,40); ax.set_ylim(-40,40)
        title += ' | all panels ego, X/Y in [-40,40] m'
    # 底部并排放置三组独立图例：来源点、框线、占据语义；不遮挡子图。
    fig.legend(handles=[Line2D([], [], linestyle='None', marker='x', color='magenta',
                               markersize=6, label='Candidate source grid\n(instance feature)')],
               title='Instance source', loc='lower center', bbox_to_anchor=(.09, .005),
               fontsize=9, frameon=True)
    fig.legend(handles=[Line2D([], [], color='lime', label='Predicted boxes'),
                        Line2D([], [], color='darkorange', label='GT boxes')],
               title='Box outlines', loc='lower center', bbox_to_anchor=(.23, .005),
               fontsize=9, frameon=True)
    bottom = .08
    projection_note = 'GT projection prioritizes associated voxels.'
    if foreground_palette is not None:
        legend = [Patch(facecolor=foreground_palette[k] / 255., label=name)
                  for k, name in FOREGROUND_CLASSES.items()]
        legend += [Patch(facecolor=str(165 / 255.), label='background occupied'),
                   Patch(facecolor='white', edgecolor='.6', label='free / ignored / masked')]
        # 12 个体素颜色项按 6 列排成两行，与来源点/框线说明分组展示。
        fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=9,
                   title='Occupancy voxel colors',
                   bbox_to_anchor=(.65, .005), frameon=True)
        projection_note = 'Occ: all foreground colored (highest per XY column); background occupied gray.'
    fig.suptitle(title + '\nRMS is feature magnitude; high score is not GT-verified quality. ' + projection_note, fontsize=11)
    fig.tight_layout(rect=[0,bottom,1,.94]); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    args = parse_args()
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
    from torchpack.utils.config import Config as TorchpackConfig
    from torch.utils.data import Subset
    from mmdet.apis import set_random_seed
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from visualize_modality_comparison import clear_pretrained, first_tensor, load_occ_colors

    set_random_seed(0, deterministic=True)
    raw_cfg = TorchpackConfig(); raw_cfg.load(str(Path(args.config).resolve()), recursive=True)
    cfg = Config(recursive_eval(raw_cfg))
    if cfg.model.heads.object.type != 'CenterHead' or cfg.model.heads.occ.type != 'BEVOCCHead2D':
        raise ValueError('仅支持 CenterHead + BEVOCCHead2D')
    dc = deepcopy(cfg.data.val); dc.test_mode = True; dc.pop('samples_per_gpu', None)
    dataset = build_dataset(dc)
    # 按验证集全局索引等间隔采样，不按场景边界重置步长。
    indices = [args.sample_index + i * args.sample_stride for i in range(args.num_samples)]
    if indices[-1] >= len(dataset):
        raise ValueError(f'请求最后一帧索引 {indices[-1]} 超出验证集范围 [0, {len(dataset)-1}]；'
                         '请减小 --sample-index、--sample-stride 或 --num-samples')
    print(f'采样验证集索引：{indices}', flush=True)
    torch.cuda.set_device(args.gpu)
    mc = deepcopy(cfg.model); clear_pretrained(mc)
    model = build_model(mc)
    load_checkpoint(model, str(Path(args.checkpoint).resolve()), map_location='cpu', strict=True)
    model.float().eval()
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    model.cuda(args.gpu)
    head = model.heads['object']; sampler = model.heads['occ'].coordinate_transform
    if sampler.point_type != 'ego' or sampler.output_h != 200 or sampler.output_w != 200 or sampler.extra_up:
        raise ValueError('仅支持当前 Occ3D ego 200×200 网格，无 extra_up')
    if list(cfg.point_cloud_range) != [-54.,-54.,-5.,54.,54.,3.]:
        raise ValueError('此可视化入口要求当前配置的源 BEV 范围')
    loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    palette = load_occ_colors(ROOT / 'tools/vis_occ3d.py')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
                  images=['overview.png', 'overview_ego.png'],
                  ego_view_extent=[-40,40,-40,40],
                  sample_index=args.sample_index, sample_stride=args.sample_stride, selected_indices=indices,
                  weight_source='checkpoint state_dict (no implicit EMA substitution)', precision='fp32',
                  score_threshold=args.score_threshold, max_instances=args.max_instances, camera_mask=args.camera_mask,
                  box_origin='geometric center', feature_definition='CenterHead.shared_conv at Top-K source index',
                  occ_bev_index_order='Y,X', gt_index_order='X,Y,Z', gt_range=[-40,-40,-1,40,40,5.4],
                  gt_projection='all valid foreground colored (highest per XY column), background occupied gray; GT boxes orange',
                  prediction_projection='all valid predicted foreground colored; selected predicted boxes green; optional camera mask',
                  first_row_background='model input LiDAR points; transformed with boxes in ego overview',
                  gt_box_source='all boxes provided by validation pipeline (dataset annotation validity filtering applies)',
                  foreground_classes=FOREGROUND_CLASSES,
                  heat='channel RMS, independent 1st/99th percentile colors', frames=[])
    def cpu(t):
        return t.detach().cpu().numpy()
    with torch.no_grad():
        for index, batch in zip(indices, loader):
            data = scatter(batch, [args.gpu])[0]
            captured = {}
            def capture(name):
                def hook(module, inputs, output):
                    captured[name] = output  # 不返回值，不改变输出
                return hook
            handles = [head.shared_conv.register_forward_hook(capture('det')),
                       head.register_forward_hook(capture('pred')),
                       sampler.register_forward_hook(capture('occ'))]
            try:
                outputs = model(**data)
            finally:
                for handle in handles:
                    handle.remove()
            metas = data['metas']
            raw = decode_candidates(head, captured['pred'])
            normal_ids = normal_selection(head, raw, metas[0]['box_type_3d'])
            # 对照原模型原始输出，防止分析分支的筛选/轴顺序悄悄偏离原实现。
            expected = outputs[0]
            bottom_boxes = raw['boxes'][normal_ids].clone()
            bottom_boxes[:,2] -= bottom_boxes[:,5] / 2
            torch.testing.assert_close(bottom_boxes.cpu(), expected['boxes_3d'].tensor.cpu(), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(raw['scores'][normal_ids].cpu(), expected['scores_3d'].cpu())
            assert torch.equal(raw['labels'][normal_ids].cpu(), expected['labels_3d'].cpu().long())
            boxes_np = cpu(raw['boxes'])
            normal_np = cpu(normal_ids)
            ids = normal_np[np.isfinite(boxes_np[normal_np]).all(-1) & (boxes_np[normal_np,3:6] > 0).all(-1)
                            & (cpu(raw['scores'])[normal_np] >= args.score_threshold)]
            transforms = tuple(cpu(first_tensor(data[k]))[0] for k in ['lidar_aug_matrix','lidar2ego','occ_aug_matrix'])
            occ_ref = cpu(sampler.ref_points)
            if not np.allclose(occ_ref.min(0)[:2], [-39.8,-39.8], atol=1e-4):
                raise ValueError('目标网格与固定 Occ3D 可视化范围不一致')
            occ_points = occ_to_lidar(occ_ref, *transforms)
            # 先剔除无有效占据 BEV 关联的框，再应用数量上限。
            source_valid = ((occ_points[:,:2] >= -54) & (occ_points[:,:2] < 54)).all(-1)
            ids = np.asarray([i for i in ids if (inside_box(occ_points, boxes_np[i]) & source_valid).any()], dtype=np.int64)
            ids = ids[np.argsort(-cpu(raw['scores'])[ids], kind='stable')][:args.max_instances]
            boxes = boxes_np[ids]
            # 一对多关联：将 ego 占据网格查询点映射到 LiDAR 系，判断其是否落在旋转框 XY 内。
            # 一个实例关联所有满足条件的有效占据 BEV 网格；这里不沿 Z 筛选单个体素。
            # 对应训练模块 BoxBEVFusion 会向这些网格传递实例消息，并结合相对几何、
            # 置信度和目标网格门控融合；重叠框的消息会聚合，并非简单复制同一个向量。
            occ_edges = make_edges(occ_points, boxes, valid=source_valid)
            det = captured['det'][0].float()
            det_flat = det.permute(1,2,0).reshape(-1, det.shape[0])
            ids_gpu = torch.as_tensor(ids, device=det.device)
            sources = raw['source'][ids_gpu]
            # 每个框按解码时保留的来源索引，直接取该检测网格的 C 维向量作为实例特征。
            # 与 ProposalHead 的提取方式一致：不是按预测框裁剪区域再做 ROIAlign/池化；
            # 下方 det_edges 仅记录框覆盖哪些检测网格，不参与这里的实例特征计算。
            feats = cpu(det_flat[sources])
            nx, ny = det.shape[1:]
            # CenterPoint coder 以第一空间轴为 X、第二轴为 Y；来源索引对应整数网格锚点。
            step = np.array(head.bbox_coder.voxel_size[:2]) * head.bbox_coder.out_size_factor
            xx, yy = np.meshgrid(np.arange(nx)*step[0]-54, np.arange(ny)*step[1]-54, indexing='ij')
            source_points = np.stack([xx, yy, np.zeros_like(xx)], -1).reshape(-1,3)
            det_edges = make_edges(source_points, boxes)
            source_xy = source_points[cpu(sources), :2]
            det_heat = cpu(det.square().mean(0).sqrt()).T
            # 捕获 CrossCoordinateSample 输出、final_conv 之前的占据 BEV 特征。
            # RMS 是各位置通道幅值的可视化；框内高亮只表示关联位置，不表示注入后的变化，
            # 本脚本仅提取实例向量并保存关联关系，不在此处调用融合模块修改占据特征。
            occ_feature = captured['occ'][0].float()
            occ_heat = cpu(occ_feature.square().mean(0).sqrt())
            sem = cpu(first_tensor(data['voxel_semantics']))
            if sem.ndim == 4: sem = sem[0]
            if sem.shape != (200,200,16): raise ValueError(f'GT shape {sem.shape}')
            valid = (sem >= 0) & (sem < 17)
            if args.camera_mask:
                camera_mask = cpu(first_tensor(data['mask_camera']))
                if camera_mask.ndim == 4: camera_mask = camera_mask[0]
                valid &= camera_mask.astype(bool)
            gx, gy, gz = np.meshgrid(np.arange(200)*.4-39.8, np.arange(200)*.4-39.8,
                                     np.arange(16)*.4-.8, indexing='ij')
            gt_points = np.stack([gx,gy,gz], -1).reshape(-1,3)
            gt_lidar = occ_to_lidar(gt_points, *transforms)
            gt_edges = make_edges(gt_lidar, boxes, True, valid.ravel())
            # GT 框直接取本帧 pipeline 输出，已与输入 LiDAR 对齐；统一为几何中心。
            gt_container = first_tensor(data['gt_bboxes_3d'])
            gt_boxes = cpu(gt_container.tensor).copy()
            gt_boxes[:, :3] = cpu(gt_container.gravity_center)
            if not np.isfinite(gt_boxes[:, :7]).all() or not (gt_boxes[:,3:6] > 0).all():
                raise ValueError('GT 框包含无效几何参数')
            all_gt_edges = make_edges(gt_lidar, gt_boxes, True, valid.ravel())
            # 全图前景着色不依赖框覆盖；关联边仍保存到 npz，供后续分析使用。
            gt_all_rgb = foreground_rgb(sem, palette, valid)
            prediction = expected['occ_pred']
            prediction = cpu(prediction) if torch.is_tensor(prediction) else np.asarray(prediction)
            if prediction.shape != sem.shape:
                raise ValueError(f'预测占据 shape {prediction.shape} 与 GT 网格不一致')
            pred_valid = (prediction >= 0) & (prediction < 17)
            if args.camera_mask:
                pred_valid &= camera_mask.astype(bool)
            # 使用预测类别自身的 occupied mask，不使用 GT 语义过滤预测结果。
            pred_edges = make_edges(gt_lidar, boxes, True, pred_valid.ravel())
            gt_rgb = foreground_rgb(prediction, palette, pred_valid)
            lidar_points = cpu(first_tensor(data['points']))[:, :3]
            token = dataset.data_infos[index]['token']
            folder = args.output_dir / f'{index:06d}_{token}'; folder.mkdir(parents=True, exist_ok=True)
            order = np.argsort(-cpu(raw['scores']), kind='stable')
            draw_result(folder/'overview.png', boxes_np[order], boxes_np[normal_np], boxes, det_heat, occ_heat,
                        source_xy, source_points, occ_points, gt_rgb, transforms, args.display_candidates, token,
                        foreground_palette=palette, gt_boxes=gt_boxes, gt_association_rgb=gt_all_rgb, lidar_points=lidar_points)
            ego_det_heat = detection_heat_in_ego(det, occ_points, occ_heat.shape, source_points)
            draw_result(folder/'overview_ego.png', boxes_np[order], boxes_np[normal_np], boxes, det_heat, occ_heat,
                        source_xy, source_points, occ_points, gt_rgb, transforms, args.display_candidates, token,
                        foreground_palette=palette, gt_boxes=gt_boxes, gt_association_rgb=gt_all_rgb,
                        ego_view=True, ego_det_heat=ego_det_heat, lidar_points=lidar_points)
            np.savez_compressed(folder/'relations.npz', raw_boxes=boxes_np, raw_scores=cpu(raw['scores']),
                raw_labels=cpu(raw['labels']), raw_source_indices=cpu(raw['source']), raw_task_ids=cpu(raw['task']),
                normal_candidate_ids=normal_np, selected_candidate_ids=ids, boxes=boxes,
                scores=cpu(raw['scores'])[ids], labels=cpu(raw['labels'])[ids], instance_feats=feats,
                source_indices=cpu(sources), source_xy=source_xy, detection_features=cpu(det),
                occupancy_bev_features=cpu(occ_feature), occ_bev_xy_lidar=occ_points,
                instance_to_detection_bev=det_edges, instance_to_occ_bev=occ_edges, instance_to_gt_voxel=gt_edges,
                lidar_points=lidar_points, gt_semantics=sem, gt_valid=valid, gt_boxes=gt_boxes,
                gt_box_labels=cpu(first_tensor(data['gt_labels_3d'])),
                gt_box_to_gt_voxel=all_gt_edges, pred_semantics=prediction, pred_valid=pred_valid,
                instance_to_pred_voxel=pred_edges, lidar_aug_matrix=transforms[0], lidar2ego=transforms[1], occ_aug_matrix=transforms[2])
            frame = dict(index=index, token=token, raw_count=len(boxes_np), normal_count=len(normal_np),
                         selected_count=len(ids), gt_box_count=len(gt_boxes),
                         gt_box_gt_edges=all_gt_edges.shape[1], prediction_edges=pred_edges.shape[1], occ_edges=occ_edges.shape[1], gt_edges=gt_edges.shape[1],
                         detector_output_verified=True, output=str(folder.resolve()))
            report['frames'].append(frame)
            (args.output_dir/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(frame), flush=True)
            del outputs, captured, data, batch


if __name__ == '__main__':
    main()
