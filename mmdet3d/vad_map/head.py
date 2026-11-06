"""直接从 DAOcc BEV 预测折线的纯卷积头，无 Query、Attention 或 Hungarian 匹配。"""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class VADVectorMapHead(nn.Module):
    """保留注册名兼容配置入口；VAD 仅指折线输出格式，不代表网络结构。"""
    def __init__(self, in_channels=512, hidden_channels=128, num_points=20,
                 num_classes=3, max_predictions=100,
                 bev_bounds=(-54.,-54.,54.,54.), map_bounds=(-30.,-15.,30.,15.),
                 heatmap_weight=1., offset_weight=1., pts_weight=1., dir_weight=.005,
                 gaussian_radius=1, collision_radius=2, nms_kernel=3, score_threshold=.05):
        super().__init__()
        if num_points<2 or num_classes<1 or max_predictions<1:
            raise ValueError('点数、类别数或最大预测数不合法')
        if nms_kernel<1 or nms_kernel%2!=1 or gaussian_radius<0 or collision_radius<0:
            raise ValueError('NMS核须为正奇数，半径须非负')
        self.num_points,self.num_classes=num_points,num_classes
        self.max_predictions,self.score_threshold=max_predictions,score_threshold
        self.gaussian_radius,self.collision_radius,self.nms_kernel=gaussian_radius,collision_radius,nms_kernel
        self.heatmap_weight,self.offset_weight=heatmap_weight,offset_weight
        self.pts_weight,self.dir_weight=pts_weight,dir_weight
        self.register_buffer('bev_bounds',torch.tensor(bev_bounds,dtype=torch.float32))
        self.register_buffer('map_bounds',torch.tensor(map_bounds,dtype=torch.float32))
        if not torch.all(self.bev_bounds[2:]>self.bev_bounds[:2]) or not torch.all(self.map_bounds[2:]>self.map_bounds[:2]):
            raise ValueError('范围须为 xmin,ymin,xmax,ymax')
        if not torch.all(self.map_bounds[:2]>=self.bev_bounds[:2]) or not torch.all(self.map_bounds[2:]<=self.bev_bounds[2:]):
            raise ValueError('地图范围必须位于BEV范围内')
        def conv(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
        self.shared=nn.Sequential(conv(in_channels,hidden_channels),conv(hidden_channels,hidden_channels))
        def branch(out):
            return nn.Sequential(conv(hidden_channels,hidden_channels),nn.Conv2d(hidden_channels,out,1))
        self.heatmap=branch(num_classes)
        # 每类独立回归，允许不同类别在同一格点各自输出一条线。
        self.offset=branch(num_classes*2)
        self.polyline=branch(num_classes*num_points*2)
        nn.init.constant_(self.heatmap[-1].bias,-2.19)

    def forward(self,bev):
        #* 保持 DAOcc [B,C,X,Y]，全程普通二维卷积，不增加 Transformer Decoder。
        with torch.cuda.amp.autocast(enabled=False):
            x=self.shared(bev.float())
            b,_,nx,ny=x.shape
            return dict(heatmap=self.heatmap(x),
                offset=self.offset(x).view(b,self.num_classes,2,nx,ny),
                points=self.polyline(x).view(b,self.num_classes,self.num_points,2,nx,ny))

    def grid(self,nx,ny,device):
        step=(self.bev_bounds[2:]-self.bev_bounds[:2])/self.bev_bounds.new_tensor([nx,ny])
        x=self.bev_bounds[0]+(torch.arange(nx,device=device)+.5)*step[0]
        y=self.bev_bounds[1]+(torch.arange(ny,device=device)+.5)*step[1]
        valid=(x[:,None]>=self.map_bounds[0])&(x[:,None]<self.map_bounds[2])&\
              (y[None]>=self.map_bounds[1])&(y[None]<self.map_bounds[3])
        return step,valid

    @torch.no_grad()
    def build_targets(self,prediction,targets):
        """按GT包围框中心分配格点；同类冲突时搜索附近空位，不进行集合匹配。"""
        heat=torch.zeros_like(prediction['heatmap'])
        b,k,nx,ny=heat.shape
        if len(targets)!=b: raise ValueError('GT数量与batch不一致')
        step,valid_grid=self.grid(nx,ny,heat.device)
        assigned=[]; moved=dropped=0
        radius=self.collision_radius
        candidates=sorted([(dx,dy) for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)],
                          key=lambda d:(d[0]**2+d[1]**2,d[0],d[1]))
        for batch,target in enumerate(targets):
            points=torch.as_tensor(target['points'],device=heat.device,dtype=torch.float32)
            labels=torch.as_tensor(target['labels'],device=heat.device,dtype=torch.long)
            if not len(labels): continue
            if labels.min()<0 or labels.max()>=k: raise ValueError('GT类别超出地图头类别范围')
            centers=(points.amin(1)+points.amax(1))*.5
            # 代表点与点序起点/方向无关。按几何顺序分配，使同一GT列表重排结果一致。
            order=sorted(range(len(labels)),key=lambda i:(int(labels[i]),*centers[i].tolist()))
            occupied=set()
            for i in order:
                label=int(labels[i]); cell=torch.floor((centers[i]-self.bev_bounds[:2])/step).long()
                ix,iy=cell.tolist(); selected=None
                for dx,dy in candidates:
                    x,y=ix+dx,iy+dy
                    if not(0<=x<nx and 0<=y<ny) or not valid_grid[x,y]: continue
                    # 同类中心至少间隔NMS半径，避免训练分配的两个正例在解码时互相抑制。
                    nr=self.nms_kernel//2
                    if any((label,x+a,y+c) in occupied for a in range(-nr,nr+1) for c in range(-nr,nr+1)): continue
                    selected=(x,y);break
                if selected is None:
                    dropped+=1;continue
                x,y=selected;occupied.add((label,x,y));moved+=int(x!=ix or y!=iy)
                origin=self.bev_bounds[:2]+step*heat.new_tensor([x+.5,y+.5])
                # 格点偏移允许超出[-.5,.5]，从而可恢复被挪到邻格的真实代表点。
                offset=(centers[i]-origin)/step
                assigned.append((batch,label,x,y,i,centers[i],offset))
                r=self.gaussian_radius
                xs=torch.arange(max(0,x-r),min(nx,x+r+1),device=heat.device)
                ys=torch.arange(max(0,y-r),min(ny,y+r+1),device=heat.device)
                gaussian=torch.exp(-((xs[:,None]-x)**2+(ys[None]-y)**2)/(2*((2*r+1)/6)**2))
                patch=heat[batch,label,xs[0]:xs[-1]+1,ys[0]:ys[-1]+1]
                patch.copy_(torch.maximum(patch,gaussian))
        return heat,assigned,valid_grid,moved,dropped

    def loss(self,prediction,targets):
        heat,assigned,valid_grid,moved,dropped=self.build_targets(prediction,targets)
        logits=prediction['heatmap'];prob=logits.sigmoid().clamp(1e-6,1-1e-6)
        # CenterNet式高斯热力图损失：峰值为正例，高斯周围降低负例惩罚。
        positive=heat.eq(1);negative=heat.lt(1)&valid_grid[None,None]
        focal=-(prob.log()*(1-prob).pow(2)*positive+
                torch.log1p(-prob)*prob.pow(2)*(1-heat).pow(4)*negative).sum()
        offset_loss=prediction['offset'].sum()*0.
        point_loss=prediction['points'].sum()*0.
        direction_loss=prediction['points'].sum()*0.
        span=self.map_bounds[2:]-self.map_bounds[:2]
        for batch,label,x,y,i,center,offset in assigned:
            offset_loss=offset_loss+F.l1_loss(prediction['offset'][batch,label,:,x,y],offset)
            pred=prediction['points'][batch,label,:,:,x,y]
            orders=torch.as_tensor(targets[batch]['orders'][i],device=pred.device,dtype=pred.dtype)
            mask=torch.as_tensor(targets[batch]['order_mask'][i],device=pred.device,dtype=torch.bool)
            relative=(orders-center)/span
            # 只在该GT的等价点序中选最优方向/起点，不在预测实例和GT间做Hungarian匹配。
            costs=(pred[None]-relative).abs().mean((-1,-2)).masked_fill(~mask,float('inf'))
            best=costs.detach().argmin()
            point_loss=point_loss+costs[best]
            dp=(pred*span).diff(dim=0);dg=orders[best].diff(dim=0)
            usable=dg.norm(dim=-1)>1e-5
            direction_loss=direction_loss+((1-F.cosine_similarity(dp,dg,dim=-1,eps=1e-6))*usable).sum()/usable.sum().clamp_min(1)
        denominator=logits.new_tensor(float(len(assigned)))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(denominator);denominator/=torch.distributed.get_world_size()
        denominator=denominator.clamp_min(1)
        return dict(heatmap=self.heatmap_weight*focal/denominator,
                    offset=self.offset_weight*offset_loss/denominator,
                    pts=self.pts_weight*point_loss/denominator,
                    dir=self.dir_weight*direction_loss/denominator,
                    num_assigned=logits.new_tensor(float(len(assigned))),
                    num_relocated=logits.new_tensor(float(moved)),
                    num_dropped=logits.new_tensor(float(dropped)))

    @torch.no_grad()
    def decode(self,prediction,augmentation):
        heat=prediction['heatmap'].sigmoid()
        b,k,nx,ny=heat.shape
        step,valid=self.grid(nx,ny,heat.device)
        heat=heat*valid[None,None]
        peaks=F.max_pool2d(heat,self.nms_kernel,stride=1,padding=self.nms_kernel//2)
        heat=heat.masked_fill(heat!=peaks,0)
        results=[];span=self.map_bounds[2:]-self.map_bounds[:2]
        for batch in range(b):
            scores,indices=heat[batch].flatten().topk(min(self.max_predictions,k*nx*ny))
            keep=(scores>=self.score_threshold)&(scores>0)
            scores,indices=scores[keep],indices[keep]
            labels=indices//(nx*ny);x=(indices%(nx*ny))//ny;y=indices%ny
            offsets=prediction['offset'][batch,labels,:,x,y]
            centers=self.bev_bounds[:2]+(torch.stack([x,y],-1)+.5+offsets)*step
            relative=prediction['points'][batch,labels,:,:,x,y]
            points=centers[:,None]+relative*span
            # 预测为增强后坐标；撤销平面BDA，沿用原矢量评估/可视化的输出协议。
            transform=augmentation[batch].float()
            points=torch.linalg.solve(transform[:2,:2],(points-transform[:2,3]).reshape(-1,2).T).T.reshape_as(points)
            results.append(dict(points=points.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy()))
        return results
