#!/usr/bin/env python3
"""CPU离线分析Occ3D三模态预测；不加载模型、不重新推理。

输入：test.py生成的<model>/<scene>/<sample_token>.pkl，以及Occ3D gts。
默认只统计mask_camera=True且GT为0..17的体素。所有比例按全数据集计数计算，
不是逐帧指标平均。混淆矩阵明确使用行=GT、列=预测，free=17。
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
           'vegetation', 'free']
MODELS = ('camera_only', 'lidar_only', 'camera_lidar_fusion')


def ratio(a, b):
    return float(a / b) if b else None


def metrics(hist):
    """保留free参与混淆计数，但语义mIoU仅平均前17类，与当前评估口径一致。"""
    tp = np.diag(hist)
    support, predicted = hist.sum(1), hist.sum(0)
    per_class = [dict(name=name, support=int(support[i]),
                      iou=ratio(tp[i], support[i]+predicted[i]-tp[i]),
                      precision=ratio(tp[i], predicted[i]), recall=ratio(tp[i], support[i]))
                 for i, name in enumerate(CLASSES)]
    ious = [v['iou'] for v in per_class[:17] if v['iou'] is not None]
    #* 所有非free类别合并为occupied；占据类别之间错分不算二分类错误。
    occupied_tp = int(hist[:17, :17].sum())
    fp, fn, tn = int(hist[17, :17].sum()), int(hist[:17, 17].sum()), int(hist[17,17])
    return dict(miou=float(np.mean(ious)) if ious else None, per_class=per_class,
                occupancy=dict(tp=occupied_tp, fp=fp, fn=fn, tn=tn,
                               iou=ratio(occupied_tp,occupied_tp+fp+fn),
                               precision=ratio(occupied_tp,occupied_tp+fp),
                               recall=ratio(occupied_tp,occupied_tp+fn)))


def transitions(joint):
    """joint[g,b,f]：GT、基线预测、融合预测的18³联合计数。

    统计基于同一体素配对，区分修复与新增错误。计数是体素数而非实例数，
    纠错比例不能直接解释为mIoU提升的贡献比例。
    """
    records = []
    for g in range(18):
        m = joint[g]
        correct_b, correct_f = int(m[g,:].sum()), int(m[:,g].sum())
        stable = int(m[g,g])
        wrong_classes = [i for i in range(17) if i != g]
        records.append(dict(
            name=CLASSES[g], support=int(m.sum()),
            baseline_correct=correct_b, fusion_correct=correct_f,
            both_correct=stable, corrected=correct_f-stable,
            newly_wrong=correct_b-stable,
            both_wrong=int(m.sum())-correct_b-correct_f+stable,
            net_correct_gain=correct_f-correct_b,
            missed_occupied_to_correct=int(m[17,g]) if g<17 else 0,
            semantic_confusion_to_correct=int(m[wrong_classes,g].sum()) if g<17 else 0,
            false_occupied_to_free=int(m[:17,17].sum()) if g==17 else 0,
            correct_to_free=int(m[g,17]) if g<17 else 0,
            correct_to_wrong_semantic=int(m[g,wrong_classes].sum()) if g<17 else 0,
            free_to_false_occupied=int(m[17,:17].sum()) if g==17 else 0,
            # 下面两项不要求语义正确，用于分析单纯占据判断的修复/退化。
            missed_occupied_to_any_occupied=int(m[17,:17].sum()) if g<17 else 0,
            occupied_to_missed=int(m[:17,17].sum()) if g<17 else 0))
    common = joint[:17,:17,:17]
    n = int(common.sum())
    return dict(per_class=records,
                totals={k:sum(r[k] for r in records) for k in records[0] if k!='name'},
                common_occupied_semantic_accuracy=dict(
                    description='GT及两个预测均为occupied的同一体素集合', count=n,
                    baseline=ratio(sum(int(common[g,g,:].sum()) for g in range(17)),n),
                    fusion=ratio(sum(int(common[g,:,g].sum()) for g in range(17)),n)))


def index_predictions(directory):
    found = {}
    for path in sorted(directory.rglob('*.pkl')):
        if path.stem in found:
            raise ValueError(f'重复token：{path.stem}')
        found[path.stem] = path
    if not found:
        raise ValueError(f'目录没有预测：{directory}')
    return found


def load_prediction(path, shape):
    # test.py的pickle包含检测框对象，因此请在原daocc环境运行以便反序列化。
    with path.open('rb') as stream:
        pred = pickle.load(stream)['occ_pred']
    if hasattr(pred, 'detach'):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred)
    if pred.shape != shape or pred.dtype.kind not in 'iu':
        raise ValueError(f'{path}形状/类型错误：{pred.shape}, {pred.dtype}')
    if np.any((pred < 0) | (pred > 17)):
        raise ValueError(f'{path}预测包含0..17之外的类别')
    return pred


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusions(output_dir):
    """读取已有18类计数绘图；无需重新加载预测、GT或模型。"""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    titles = ('Camera only', 'LiDAR only', 'Camera + LiDAR')
    matrices = []
    for name in MODELS:
        hist = np.load(output_dir / (name + '_confusion.npy'), allow_pickle=False)
        if hist.shape != (18, 18) or np.any(hist < 0):
            raise ValueError(f'{name}混淆计数应为非负18×18矩阵')
        #* 按GT行归一化：每格是该GT类别被预测为列类别的比例。
        # 保留free列以观察占据漏检；对角线是类别Recall，而不是IoU。
        total = hist.sum(1, keepdims=True)
        matrices.append(np.divide(hist * 100., total,
                         out=np.zeros((18,18)), where=total > 0))
    plot_dir = output_dir / 'figures'
    plot_dir.mkdir(parents=True, exist_ok=True)

    def draw(ax, matrix, title, difference=False, limit=100., fontsize=7):
        image = ax.imshow(matrix, cmap='RdBu_r' if difference else 'Blues',
                          vmin=-limit if difference else 0, vmax=limit,
                          interpolation='nearest')
        ax.set_xticks(range(18))
        ax.set_yticks(range(18))
        ax.set_xticklabels(CLASSES, rotation=55, ha='right', fontsize=fontsize+1)
        ax.set_yticklabels(CLASSES, fontsize=fontsize+1)
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('Ground-truth class')
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(18)-.5, minor=True)
        ax.set_yticks(np.arange(18)-.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=.4)
        ax.tick_params(which='minor', bottom=False, left=False)
        # free行/列与其他语义类分开；所有图保持完全相同的类别顺序。
        ax.axhline(16.5, color='gray', linewidth=1)
        ax.axvline(16.5, color='gray', linewidth=1)
        for i in range(18):
            for j in range(18):
                v = matrix[i,j]
                # 隐去近零数值减少拥挤；仍保留对角线数值。
                if abs(v) >= .1 or i == j:
                    label = f'{v:+.1f}' if difference else f'{v:.1f}'
                    color = 'white' if abs(v) > limit*.55 else 'black'
                    ax.text(j,i,label,ha='center',va='center',fontsize=fontsize,color=color)
        return image

    for name,title,matrix in zip(MODELS,titles,matrices):
        fig,ax = plt.subplots(figsize=(12,10), constrained_layout=True)
        im = draw(ax,matrix,title + ' — row-normalized (%)')
        fig.colorbar(im,ax=ax,shrink=.8,label='Percent of GT class (%)')
        fig.savefig(plot_dir/(name+'_confusion.png'),dpi=180)
        plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(30,10),constrained_layout=True)
    for ax,title,matrix in zip(axes,titles,matrices):
        im = draw(ax,matrix,title,fontsize=6)
    fig.colorbar(im,ax=list(axes),shrink=.6,label='Percent of GT class (%)')
    fig.suptitle('Shared scale 0–100% | diagonal = class recall | free column = missed occupancy',fontsize=15)
    fig.savefig(plot_dir/'confusion_comparison.png',dpi=180)
    plt.close(fig)

    #* 差值是行归一化比例的百分点差，不能直接当作逐体素纠错数量。
    # 红色表示该预测去向增多；对角线上红色通常为改善，非对角线上蓝色通常为改善。
    delta = matrices[2] - matrices[1]
    fig,ax = plt.subplots(figsize=(12,10),constrained_layout=True)
    im = draw(ax,delta,'Fusion minus LiDAR — percentage points',True,
              max(float(np.abs(delta).max()),.1))
    fig.colorbar(im,ax=ax,shrink=.8,label='Change (percentage points)')
    fig.savefig(plot_dir/'fusion_minus_lidar_confusion.png',dpi=180)
    plt.close(fig)
    print('混淆矩阵图片保存至',plot_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-root', type=Path, default=Path('/home/wangyushen/Data/Outputs/daocc/occ_predictions'))
    parser.add_argument('--gt-root', type=Path, default=Path('/home/wangyushen/Data/Datasets/nuscenes_anns/gts'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, help='仅测试前N个匹配token，省略则分析全部')
    parser.add_argument('--plot-only', action='store_true',
                        help='仅读取output-dir内已有confusion.npy生成图片，不重算预测')
    args = parser.parse_args()
    if args.plot_only:
        plot_confusions(args.output_dir)
        return
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('max-frames必须大于0')
    paths = {name:index_predictions(args.pred_root/name) for name in MODELS}
    tokens = set(paths[MODELS[0]])
    for name in MODELS[1:]:
        if set(paths[name]) != tokens:
            raise ValueError(f'{name}与camera_only的token集合不一致，请补全预测后再分析')
    selected = sorted(tokens)
    if args.max_frames:
        selected = selected[:args.max_frames]
    #* 预先校验全部GT路径和场景归属，避免处理一半才发现缺失文件。
    gt_paths = {}
    for token in selected:
        scene = paths[MODELS[0]][token].parent.name
        if any(paths[name][token].parent.name != scene for name in MODELS):
            raise ValueError(f'{token}场景目录不一致')
        p = args.gt_root/scene/token/'labels.npz'
        if not p.is_file():
            raise FileNotFoundError(p)
        gt_paths[token] = p
    histograms = {name:np.zeros((18,18),dtype=np.int64) for name in MODELS}
    joint_counts = {name:np.zeros((18,18,18),dtype=np.int64) for name in MODELS[:2]}
    frame_rows = []
    print(f'三组预测各{len(tokens)}帧，本次分析{len(selected)}帧；CPU、mask_camera过滤。',flush=True)
    for index, token in enumerate(selected):
        with np.load(gt_paths[token]) as labels:
            gt = labels['semantics']
            mask = labels['mask_camera'].astype(bool)
        if gt.shape != (200,200,16) or mask.shape != gt.shape:
            raise ValueError(f'{token}不是预期的Occ3D网格')
        if np.any((gt != 255) & ((gt < 0) | (gt > 17))):
            raise ValueError(f'{token} GT含未知类别')
        valid = mask & (gt >= 0) & (gt < 18)
        g = gt[valid].astype(np.int64)
        predictions = {name:load_prediction(paths[name][token],gt.shape)[valid].astype(np.int64)
                       for name in MODELS}
        row = dict(token=token,scene=gt_paths[token].parents[1].name,valid_voxels=len(g),
                   masked_voxels=int(mask.sum()),ignored_masked_gt=int((mask & ~valid).sum()))
        for name, pred in predictions.items():
            h = np.bincount(g*18+pred,minlength=18*18).reshape(18,18)
            histograms[name] += h
            row[name+'_correct'] = int(np.trace(h))
        f = predictions['camera_lidar_fusion']
        for name in joint_counts:
            joint_counts[name] += np.bincount(g*324+predictions[name]*18+f,
                                             minlength=18**3).reshape(18,18,18)
        frame_rows.append(row)
        if (index+1)%100 == 0 or index+1 == len(selected):
            print(f'[{index+1}/{len(selected)}] {token}',flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report = dict(frames=len(selected),available_prediction_frames=len(tokens),
                  partial=len(selected)!=len(tokens),pred_root=str(args.pred_root.resolve()),
                  gt_root=str(args.gt_root.resolve()),classes=CLASSES,
                  mask='mask_camera & valid GT 0..17; GT255 ignored',
                  matrix_axes='row=GT, column=prediction; joint axes=GT,baseline,fusion',
                  metric_units='ratios 0..1; multiply by 100 for percent',
                  models={},comparisons={})
    for name,h in histograms.items():
        report['models'][name] = metrics(h)
        np.save(args.output_dir/(name+'_confusion.npy'),h)
        normalized = np.divide(h,h.sum(1,keepdims=True),out=np.zeros_like(h,dtype=float),where=h.sum(1,keepdims=True)>0)
        for suffix,matrix in [('counts',h),('row_normalized',normalized)]:
            save_csv(args.output_dir/f'{name}_confusion_{suffix}.csv',
                     [dict(gt=CLASSES[i],**{c:matrix[i,j].item() for j,c in enumerate(CLASSES)}) for i in range(18)])
        save_csv(args.output_dir/(name+'_metrics.csv'),report['models'][name]['per_class'])
    for name,joint in joint_counts.items():
        comparison = transitions(joint)
        report['comparisons']['fusion_vs_'+name] = comparison
        np.save(args.output_dir/('fusion_vs_'+name+'_joint_counts.npy'),joint)
        save_csv(args.output_dir/('fusion_vs_'+name+'_transitions.csv'),comparison['per_class'])
    save_csv(args.output_dir/'frames.csv',frame_rows)
    (args.output_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,result in report['models'].items():
        print(name,'mIoU =',None if result['miou'] is None else round(100*result['miou'],3),
              'occupancy =',result['occupancy'])
    print('结果保存至',args.output_dir.resolve())
    plot_confusions(args.output_dir)


if __name__ == '__main__':
    main()
