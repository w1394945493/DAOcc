"""Offline GT-object analysis of saved DAOcc detection and Occ3D predictions.

This is a local diagnostic, not the official nuScenes detection evaluator.
Only trusted local pickle files should be used. No model or GPU is required.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_training_sampling import load_config

OCC_CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
OCC_MIN = np.array([-40., -40., -1.])
VOXEL = .4
SHAPE = (200, 200, 16)


def array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def match_boxes(gt, names, pred, labels, scores, classes, threshold, score_min):
    """Score-ordered same-class one-to-one XY center matching in LiDAR coordinates."""
    matches = {}
    unmatched = []
    for j in np.argsort(-scores, kind='stable'):
        if scores[j] < score_min:
            continue
        candidates = [i for i, name in enumerate(names) if name == classes[labels[j]] and i not in matches]
        if candidates:
            distances = np.linalg.norm(gt[candidates, :2] - pred[j, :2], axis=1)
            k = int(np.argmin(distances))
            if distances[k] <= threshold:
                matches[candidates[k]] = int(j)
                continue
        unmatched.append(int(j))
    return matches, unmatched


def box_region(box, rotation, translation):
    """OCC ego voxel centers -> LiDAR -> yaw-aligned box; full lidar2ego rotation."""
    c, s = np.cos(box[6]), np.sin(box[6])
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    corners = np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    corners = (corners * box[3:6]) @ yaw.T + box[:3]
    ego = corners @ rotation.T + translation
    lo = np.maximum(np.floor((ego.min(0)-OCC_MIN)/VOXEL).astype(int), 0)
    hi = np.minimum(np.ceil((ego.max(0)-OCC_MIN)/VOXEL).astype(int), SHAPE)
    if np.any(hi <= lo):
        return None
    ijk = np.stack(np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)), indexing='ij'), -1)
    xyz = OCC_MIN + (ijk + .5)*VOXEL
    lidar = (xyz - translation) @ rotation
    local = (lidar - box[:3]) @ yaw
    inside = (np.abs(local) <= box[3:6]/2 + 1e-6).all(-1)
    return tuple(slice(int(a), int(b)) for a,b in zip(lo,hi)), inside, ego


def summarize(rows, field):
    result = []
    for key in sorted(set(r[field] for r in rows)):
        group = [r for r in rows if r[field] == key]
        valid = [r for r in group if r['occ_evaluable']]
        denom = sum(r['target_voxels'] for r in valid)
        result.append(dict(group=key, objects=len(group), matched=sum(r['matched'] for r in group),
                           occ_evaluable=len(valid),
                           mean_object_recall=float(np.mean([r['occ_recall'] for r in valid])) if valid else None,
                           voxel_weighted_recall=sum(r['correct_voxels'] for r in valid)/denom if denom else None,
                           mean_local_iou=float(np.mean([r['local_iou'] for r in valid])) if valid else None))
    return result


def render(path, row, gt, pred, valid, region, corners, pred_corners):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sl, inside, _ = region
    target = OCC_CLASSES.index(row['class_name'])
    g, p, v = gt[sl], pred[sl], valid[sl] & inside
    extent = [OCC_MIN[0]+sl[0].start*VOXEL, OCC_MIN[0]+sl[0].stop*VOXEL,
              OCC_MIN[1]+sl[1].start*VOXEL, OCC_MIN[1]+sl[1].stop*VOXEL]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    # Convex hull projection accounts for lidar-to-ego roll/pitch as well as yaw.
    from scipy.spatial import ConvexHull
    for pts, color, label in [(corners, 'green', 'GT'), (pred_corners, 'red', 'prediction')]:
        if pts is not None:
            hull = ConvexHull(pts[:, :2]); ids = list(hull.vertices)+[hull.vertices[0]]
            axes[0].plot(pts[ids,0],pts[ids,1], color=color,label=label)
    axes[0].legend(); axes[0].set_title('Boxes in ego BEV')
    for ax, data, title in zip(axes[1:], [(v & (g==target)).sum(2), (v & (p==target)).sum(2),
                                        (v & (g==target) & (p!=g)).sum(2)],
                              ['GT target voxel count', 'Predicted target count', 'Missed / wrong target count']):
        im=ax.imshow(data.T,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=SHAPE[2]);fig.colorbar(im,ax=ax)
        ax.set_title(title)
    for ax in axes: ax.set_aspect('equal');ax.set_xlabel('Ego X (m)');ax.set_ylabel('Ego Y (m)')
    fig.suptitle('%s | %s | GT %s | %s\nrecall=%.3f IoU=%.3f valid target=%d; masked GT-box region only' %
                 (row['sample_token'],row['class_name'],row['gt_index'],row['error_group'],row['occ_recall'],row['local_iou'],row['target_voxels']))
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--pred-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--match-distance',type=float,default=2.)
    parser.add_argument('--score-threshold',type=float,default=.1)
    parser.add_argument('--min-target-voxels',type=int,default=5)
    parser.add_argument('--poor-recall',type=float,default=.5)
    parser.add_argument('--good-recall',type=float,default=.8)
    parser.add_argument('--max-cases-per-group',type=int,default=20)
    parser.add_argument('--max-frames',type=int,default=0,help='Debug limit; 0 analyzes all available frames.')
    args=parser.parse_args()
    if not (0 <= args.poor_recall < args.good_recall <= 1 and 0 <= args.score_threshold <= 1): parser.error('Invalid thresholds')
    if args.match_distance<=0 or args.min_target_voxels<1 or min(args.max_frames,args.max_cases_per_group)<0: parser.error('Invalid counts/distance')
    cfg=load_config(args.config)
    dc=cfg.data.test
    if dc.type!='NuScenesDatasetOccupancy' or dc.data_type!='occ3d': raise ValueError('Only standard NuScenesDatasetOccupancy / Occ3D supported')
    # Explicitly reject inference augmentations incompatible with unaugmented saved boxes.
    for t in dc.pipeline:
        if t.type in ('RandomFlip3D','MultiScaleFlipAug3D'): raise ValueError('Augmented test pipeline unsupported')
        if t.type in ('ImageAug3D','GlobalRotScaleTrans') and t.get('is_train',False): raise ValueError('Training augmentation in test pipeline')
    with open(dc.ann_file,'rb') as f: infos=sorted(pickle.load(f)['infos'],key=lambda i:i['timestamp'])
    infos=infos[::dc.get('load_interval',1)]
    classes=list(dc.get("classes", cfg.object_classes))
    files={}
    for p in args.pred_dir.rglob('*.pkl'):
        if p.stem in files: raise ValueError('Duplicate sample token: '+p.stem)
        files[p.stem]=p
    missing=[i['token'] for i in infos if i['token'] not in files]
    if missing and not args.allow_partial: raise ValueError('%d predictions missing; use --allow-partial during inference'%len(missing))
    selected=[i for i in infos if i['token'] in files]
    if args.max_frames: selected=selected[:args.max_frames]
    if not selected: raise ValueError('No matching predictions')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if any(args.out_dir.iterdir()): raise ValueError('Use an empty output directory to avoid mixing analyses')
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(occ_classes=OCC_CLASSES,occ_min=OCC_MIN.tolist(),voxel_size=VOXEL,shape=SHAPE,
                    matching='greedy score order, same class, LiDAR XY center distance; NOT official nuScenes AP',
                    scope='mask_camera and non-ignore voxels inside GT box; overlapping boxes may count the same voxel; no instance OCC labels',
                    case_selection='first qualifying objects in timestamp order, not a random sample')
    write_json(args.out_dir/'settings.json',settings)
    rows=[]; false_positives=[];cases={};unmapped=0
    from pyquaternion import Quaternion
    for frame_no,info in enumerate(selected):
        token=info['token']
        with files[token].open('rb') as f: result=pickle.load(f)
        for key in ['boxes_3d','scores_3d','labels_3d','occ_pred']:
            if key not in result: raise ValueError('%s missing %s'%(token,key))
        boxes=result['boxes_3d']
        if type(boxes).__name__!='LiDARInstance3DBoxes': raise ValueError('Expected LiDARInstance3DBoxes')
        pred=array(boxes.tensor).copy();pred[:,:3]=array(boxes.gravity_center)
        scores=array(result['scores_3d']);labels=array(result['labels_3d']).astype(int)
        if not np.isfinite(pred[:,:7]).all() or not np.isfinite(scores).all():raise ValueError('Nonfinite detection: '+token)
        if np.any(labels<0) or np.any(labels>=len(classes)):raise ValueError('Invalid detection labels')
        occ=array(result['occ_pred'])
        with np.load(Path(info['occ3d']['occ_path'])/'labels.npz') as data:
            gt=data['semantics'];valid=data['mask_camera'].astype(bool)&(gt!=255)
        if occ.shape!=SHAPE or gt.shape!=SHAPE:raise ValueError('Expected Occ3D [X,Y,Z]=(200,200,16)')
        if not np.isin(occ,np.arange(18)).all():raise ValueError('Invalid OCC predictions')
        keep=np.asarray(info['valid_flag'] if dc.get('use_valid_flag',False) else np.asarray(info['num_lidar_pts'])>0,dtype=bool)
        keep &= np.isin(info['gt_names'],classes)
        ids=np.flatnonzero(keep);gtboxes=np.asarray(info['gt_boxes'])[keep];names=np.asarray(info['gt_names'])[keep]
        matched,unmatched=match_boxes(gtboxes,names,pred,labels,scores,classes,args.match_distance,args.score_threshold)
        for j in unmatched:false_positives.append(dict(sample_token=token,pred_index=j,class_name=classes[labels[j]],score=float(scores[j])))
        rotation=Quaternion(info['lidar2ego_rotation']).rotation_matrix;translation=np.asarray(info['lidar2ego_translation'])
        for i,(box,name) in enumerate(zip(gtboxes,names)):
            if name not in OCC_CLASSES:unmapped+=1;continue
            j=matched.get(i);distance=float(np.linalg.norm((box[:3]@rotation.T+translation)[:2]))
            row=dict(sample_token=token,gt_index=int(ids[i]),class_name=str(name),distance_m=distance,
                     distance_bin='0-20' if distance<20 else '20-40' if distance<40 else '40+',matched=j is not None,
                     pred_index=j,score=float(scores[j]) if j is not None else None,
                     center_error_m=float(np.linalg.norm(box[:2]-pred[j,:2])) if j is not None else None,
                     size_l1_m=float(np.abs(box[3:6]-pred[j,3:6]).mean()) if j is not None else None,
                     yaw_error_rad=float(abs((box[6]-pred[j,6]+np.pi)%(2*np.pi)-np.pi)) if j is not None else None,
                     valid_region_voxels=0,target_voxels=0,correct_voxels=0,occ_evaluable=False,
                     occ_recall=None,empty_ratio=None,wrong_class_ratio=None,local_iou=None,error_group='not_evaluable',confusion='{}')
            region=box_region(box,rotation,translation)
            if region:
                sl,inside,_=region;v=valid[sl]&inside;g=gt[sl];p=occ[sl];cls=OCC_CLASSES.index(name)
                target=v&(g==cls);n=int(target.sum());tp=int((target&(p==cls)).sum());union=int((v&((g==cls)|(p==cls))).sum())
                row.update(valid_region_voxels=int(v.sum()),target_voxels=n,correct_voxels=tp)
                if n>=args.min_target_voxels:
                    counts=np.bincount(p[target].astype(int),minlength=18)
                    row.update(occ_evaluable=True,occ_recall=tp/n,empty_ratio=int(counts[17])/n,
                               wrong_class_ratio=(n-tp-int(counts[17]))/n,local_iou=tp/union,
                               confusion=json.dumps({OCC_CLASSES[k]:int(x) for k,x in enumerate(counts) if x},ensure_ascii=False))
                    quality='poor' if tp/n<args.poor_recall else 'good' if tp/n>=args.good_recall else 'intermediate'
                    row['error_group']=('matched' if j is not None else 'missed')+'_occ_'+quality
                    group=row['error_group']
                    if group in ['matched_occ_poor','missed_occ_good','missed_occ_poor'] and cases.get(group,0)<args.max_cases_per_group:
                        folder=args.out_dir/'cases'/group;folder.mkdir(parents=True,exist_ok=True)
                        other=box_region(pred[j],rotation,translation) if j is not None else None
                        render(folder/('%s_gt%d.png'%(token,ids[i])),row,gt,occ,valid,region,region[2],other[2] if other else None)
                        cases[group]=cases.get(group,0)+1
            rows.append(row)
        if (frame_no+1)%100==0:print('Analyzed %d/%d frames'%(frame_no+1,len(selected)),flush=True)
    if rows:
        write_csv(args.out_dir/'objects.csv',rows)
        for field,filename in [('class_name','summary_by_class'),('distance_bin','summary_by_distance'),('error_group','summary_by_group')]:
            write_csv(args.out_dir/(filename+'.csv'),summarize(rows,field))
    write_csv(args.out_dir/'unmatched_predictions.csv',false_positives,['sample_token','pred_index','class_name','score'])
    write_json(args.out_dir/'coverage.json',dict(dataset_frames=len(infos),available_frames=len(infos)-len(missing),
        analyzed_frames=len(selected),missing_tokens=missing,objects=len(rows),occ_evaluable=sum(r['occ_evaluable'] for r in rows),
        unmapped_objects=unmapped,cases=cases))
    print('Saved:',args.out_dir)


if __name__=='__main__':main()
