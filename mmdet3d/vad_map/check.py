"""CPU 自检，不需要 nuScenes 数据或 GPU：python -m mmdet3d.vad_map.check。"""
import numpy as np
import torch
from torch import nn
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree
from unittest.mock import patch
from .head import VADVectorMapHead
from .geometry import VectorMapExtractor, equivalent_orders
from .evaluation import evaluate_vectors
from .model import DAOccVADMap
from mmdet3d.models.fusion_models.bevfusion import BEVFusion


def main():
    torch.manual_seed(3); torch.set_num_threads(1)
    #* 构造模拟地图：道路面大于窗口，窗口边缘不应被错误标为道路边界。
    extractor=VectorMapExtractor('/unused',bounds=(-4,-3,4,3),num_points=5)
    divider=LineString([(-8,0),(8,0)])
    crossing=Polygon([(-1,-1),(1,-1),(1,1),(-1,1)])
    road=Polygon([(-20,-20),(20,-20),(20,20),(-20,20)])
    def index(gs): return gs,STRtree(gs) if gs else None
    extractor._locations['fake']={'road_divider':index([divider]),'lane_divider':index([]),
          'ped_crossing':index([crossing]),'road_segment':index([road]),'lane':index([])}
    info=dict(ego2global_rotation=[1,0,0,0],lidar2ego_rotation=[1,0,0,0],
              ego2global_translation=[0,0,0],lidar2ego_translation=[0,0,0])
    gt=extractor(info,'fake')
    assert set(gt['labels'])=={0,1},gt['labels']
    assert gt['order_mask'][gt['labels']==1].sum()==8
    aug=np.eye(4); aug[:2,:2]=[[0,-1],[1,0]]
    rotated=extractor(info,'fake',aug)
    line=rotated['points'][rotated['labels']==0][0]
    assert np.allclose(line[:,0],0) and np.allclose(np.sort(line[:,1])[[0,-1]],[-3,3])
    print('PASS: 地图裁剪、闭环等价序、旋转增强、无虚假窗口边界')
    #* 非正方形BEV、多类回归及空GT批次的梯度检查。
    head=VADVectorMapHead(in_channels=8,hidden_channels=16,num_points=5,
         map_bounds=(-4,-3,4,3),bev_bounds=(-4,-3,4,3),max_predictions=10)
    empty={k:v[:0] for k,v in gt.items()}
    bev=torch.randn(2,8,8,6,requires_grad=True)
    outputs=head(bev); losses=head.loss(outputs,[gt,empty])
    sum(v for k,v in losses.items() if not k.startswith('num_')).backward()
    assert torch.isfinite(bev.grad).all() and bev.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    zero_loss=head.loss(head(bev.detach()),[empty,empty])
    assert all(torch.isfinite(v) for v in zero_loss.values())
    assert not any(isinstance(m,nn.MultiheadAttention) for m in head.modules())
    print('PASS: 纯卷积前向反向、空GT、全部参数及共享BEV梯度')
    #* 根据GT直接填入完美的中心偏移/点偏移，验证编码解码及点序选择。
    prediction={k:torch.zeros_like(v[:1]) for k,v in outputs.items()}
    prediction['heatmap'].fill_(-20.)
    _,assigned,_,_,_=head.build_targets(prediction,[gt])
    assert len(assigned)==len(gt['labels'])
    for batch,label,x,y,i,center,offset in assigned:
        prediction['heatmap'][batch,label,x,y]=20.
        prediction['offset'][batch,label,:,x,y]=offset
        prediction['points'][batch,label,:,:,x,y]=(torch.from_numpy(gt['points'][i][::-1].copy())-center)/(head.map_bounds[2:]-head.map_bounds[:2])
    loss=head.loss(prediction,[gt])
    assert loss['pts'].item()<1e-6 and loss['offset'].item()<1e-6
    decoded=head.decode(prediction,torch.eye(4)[None])[0]
    report=evaluate_vectors([decoded],[gt],extractor.classes)
    assert abs(report['vector_map/mAP']-1)<1e-6
    # 撤销旋转+平移，解码结果应满足同样的矩阵关系。
    augmentation=torch.eye(4)[None];augmentation[0,:2,:2]=torch.tensor([[0.,-2.],[2.,0.]])
    augmentation[0,:2,3]=torch.tensor([1.,-1.])
    restored=head.decode(prediction,augmentation)[0]['points']
    reproj=restored@augmentation[0,:2,:2].numpy().T+augmentation[0,:2,3].numpy()
    assert np.allclose(reproj,decoded['points'],atol=1e-6)
    print('PASS: XY网格、正反点序、偏移编码解码、增强恢复、完美AP=1')
    #* 同类中心重叠时不会覆盖回归GT；移动中心仍能准确恢复原折线。
    duplicate={k:np.concatenate([v,v],axis=0) for k,v in gt.items()}
    _,slots,_,moved,dropped=head.build_targets(prediction,[duplicate])
    keys=[tuple(t[:4]) for t in slots]
    assert len(keys)==len(set(keys)) and moved>0
    small=VADVectorMapHead(in_channels=8,hidden_channels=8,num_points=5,collision_radius=0,
                          map_bounds=(-4,-3,4,3),bev_bounds=(-4,-3,4,3))
    _,_,_,_,dropped=small.build_targets(prediction,[duplicate])
    assert dropped>0
    print('PASS: 同类中心冲突重定位及无法分配计数')
    #* 派生模型调用原前向一次，保留原loss并增加地图loss，hook及时移除。
    model=DAOccVADMap.__new__(DAOccVADMap); nn.Module.__init__(model)
    model.decoder=nn.ModuleDict({'neck':nn.Identity()});model.vector_map_head=head
    model.vector_map_loss_weight=1.
    def parent(self,img,*args,**kw):
        x=self.decoder['neck'](img)
        return {'loss/occ/mock':x.square().mean()} if self.training else [{} for _ in x]
    inputs=dict(img=bev,points=None,camera2ego=None,lidar2ego=None,lidar2camera=None,
       lidar2image=None,camera_intrinsics=None,camera2lidar=None,img_aug_matrix=None,
       lidar_aug_matrix=torch.eye(4)[None].repeat(2,1,1),metas=[{},{}])
    with patch.object(BEVFusion,'forward_single',parent):
        model.train(); out=model.forward_single(**inputs,gt_vector_map=[gt,empty])
        assert 'loss/occ/mock' in out and 'loss/vector_map/heatmap' in out
        model.eval(); out=model.forward_single(**inputs)
        assert all('vector_map' in sample for sample in out)
    assert not model.decoder['neck']._forward_hooks
    print('PASS: 派生模型训练/推理接入、hook清理、原任务结果保留')


if __name__=='__main__': main()
