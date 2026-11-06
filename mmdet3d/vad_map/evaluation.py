"""矢量地图的 Chamfer AP；独立实现，非 VAD 官方评估结果的逐位复刻。"""
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from .geometry import sample_line


def chamfer(a, b):
    distance = cdist(a,b)
    return (distance.min(0).mean()+distance.min(1).mean())/2


def average_precision(tp, fp, total):
    if total == 0:
        return float('nan')
    if len(tp) == 0:
        return 0.
    recall = np.cumsum(tp)/total
    precision = np.cumsum(tp)/np.maximum(np.cumsum(tp)+np.cumsum(fp),1)
    recall = np.r_[0.,recall,1.]
    precision = np.r_[0.,precision,0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.flatnonzero(recall[1:] != recall[:-1])
    return float(((recall[indices+1]-recall[indices])*precision[indices+1]).sum())


def evaluate_vectors(predictions, targets, classes, thresholds=(.5,1.,1.5), sample_points=100):
    metrics, class_aps = {}, []
    for label, name in enumerate(classes):
        gt, detections, distances = {}, [], {}
        for frame, (prediction,target) in enumerate(zip(predictions,targets)):
            gt[frame] = [sample_line(LineString(points),sample_points) for points,cat in
                         zip(target['points'],target['labels']) if cat == label]
            for idx,(points,score,cat) in enumerate(zip(prediction['points'],prediction['scores'],prediction['labels'])):
                if cat != label: continue
                if not np.isfinite(points).all() or not np.isfinite(score):
                    raise ValueError('地图预测包含非有限值')
                curve = sample_line(LineString(points),sample_points)
                distances[(frame,idx)] = np.array([chamfer(curve,g) for g in gt[frame]])
                detections.append((float(score),frame,idx))
        detections.sort(key=lambda d: d[0],reverse=True)
        total = sum(map(len,gt.values()))
        aps = []
        for threshold in thresholds:
            matched = {i:np.zeros(len(g),dtype=bool) for i,g in gt.items()}
            tp,fp=[],[]
            for _,frame,idx in detections:
                distance = distances[(frame,idx)]
                # 每个预测匹配最近 GT；重复命中同一 GT 计为 FP。
                best = int(distance.argmin()) if len(distance) else -1
                valid = best >= 0 and distance[best] <= threshold and not matched[frame][best]
                if valid: matched[frame][best]=True
                tp.append(int(valid)); fp.append(int(not valid))
            ap=average_precision(tp,fp,total)
            metrics[f'vector_map/{name}/AP@{threshold:g}m']=ap
            aps.append(ap)
        mean=float(np.mean(aps))
        metrics[f'vector_map/{name}/AP']=mean
        metrics[f'vector_map/{name}/num_gt']=total
        if np.isfinite(mean): class_aps.append(mean)
    metrics['vector_map/mAP']=float(np.mean(class_aps)) if class_aps else float('nan')
    return metrics
