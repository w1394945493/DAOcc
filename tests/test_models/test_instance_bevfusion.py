"""Run: python -m unittest discover -s tests/test_models -p test_instance_bevfusion.py"""
import unittest
from copy import deepcopy
import torch
from torch import nn
from mmcv import Config
from torchpack.utils.config import Config as TorchConfig
from mmdet3d.models import build_head
from mmdet3d.models.customs.box_bev_fusion import BoxBEVFusion
from mmdet3d.models.fusion_models.instance_bevfusion import InstanceBEVFusion
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.utils import recursive_eval


class TestBoxFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.fusion = BoxBEVFusion(4, 8, max_instances=2)
        xy = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [-1., 1., 0.], [1., 1., 0.]])
        self.points = xy[None].repeat(2, 1, 1)
        self.bounds = (-10, -10, 10, 10)

    def item(self, empty=False):
        n = 0 if empty else 1
        return dict(features=torch.ones(n, 4, requires_grad=True),
                    boxes=torch.tensor([[0., 0., 0., 4., 4., 2., 0., 0., 0.]])[:n],
                    scores=torch.full((n,), .8), labels=torch.zeros(n, dtype=torch.long))

    def test_batch_isolation_and_gradients(self):
        bev = torch.randn(2, 8, 2, 2, requires_grad=True)
        items = [self.item(), self.item(True)]
        result = self.fusion(bev, items, self.points, self.bounds)
        self.assertTrue(torch.equal(result[1], bev[1]))
        alone = self.fusion(bev[:1], items[:1], self.points[:1], self.bounds)
        torch.testing.assert_close(alone[0], result[0])
        result.square().sum().backward()
        self.assertGreater(items[0]['features'].grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.fusion.parameters()))

    def test_empty_ddp_graph_and_outside(self):
        bev = torch.randn(2, 8, 2, 2, requires_grad=True)
        items = [self.item(True), self.item()]
        items[1]['boxes'][:, :2] = 100
        result = self.fusion(bev, items, self.points, self.bounds)
        self.assertTrue(torch.equal(result, bev))
        result.sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() == 0 for p in self.fusion.parameters()))

    def test_rotation_and_overlap_normalization(self):
        item = self.item()
        item['boxes'][0, 3:5] = torch.tensor([4., 1.])
        item['boxes'][0, 6] = torch.pi / 2
        pts = torch.tensor([[0., 1.5, 0.], [1.5, 0., 0.]])
        src, dst, _ = self.fusion.build_edges(pts, item, self.bounds)
        self.assertEqual(dst.tolist(), [0])
        # 相同内容、相同几何的重复实例，在权重饱和后不应加倍更新。
        item = self.item(); item['scores'].fill_(1)
        twice = {k: torch.cat([v, v]) for k, v in item.items()}
        bev = torch.randn(1,8,2,2)
        a = self.fusion(bev, [item], self.points[:1], self.bounds)
        b = self.fusion(bev, [twice], self.points[:1], self.bounds)
        torch.testing.assert_close(a,b)

    def test_optional_attention_aggregation(self):
        fusion = BoxBEVFusion(4, 8, max_instances=2, aggregation='attention',
                              attention_channels=4)
        item = self.item()
        twice = {k: torch.cat([v, v]) for k, v in item.items()}
        bev = torch.randn(1, 8, 2, 2, requires_grad=True)
        single = fusion(bev, [item], self.points[:1], self.bounds)
        duplicate = fusion(bev, [twice], self.points[:1], self.bounds)
        # 完全相同实例的注意力权重均分，聚合结果不应因重复而加倍。
        torch.testing.assert_close(single, duplicate)
        single.square().sum().backward()
        self.assertIsNotNone(fusion.attention_query.weight.grad)
        self.assertIsNotNone(fusion.attention_key.weight.grad)

    def test_query_transform_matches_sampler(self):
        # 非方形网格、多BS、不同外参/增强；用实际采样器验证查询坐标。
        from mmdet3d.models.customs.cross_coordinate_sample import CrossCoordinateSample
        sampler = CrossCoordinateSample([-2,2,-3,3,0,0], [4,6,1], [-10,10,-10,10], in_dim=1,out_dim=1)
        la = torch.eye(4)[None].repeat(2,1,1); le=la.clone(); oa=la.clone()
        le[0,:3,3] = torch.tensor([.2,-.3,.5])
        le[1,:2,:2] = torch.tensor([[0.,-1.],[1.,0.]])
        la[1,:3,:3] *= 1.1; la[1,0,3] = .4
        oa[1,0,0] = -1
        points = self.fusion.query_points(sampler,la,le,oa)
        x = torch.arange(20).float()-9.5
        image = x[None,None,None,:].expand(2,1,20,20).clone()
        sampled = sampler(image,la,le,oa).flatten(1)
        torch.testing.assert_close(sampled,points[...,0],atol=2e-6,rtol=1e-5)


class TestHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c=TorchConfig();c.load('configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_instance_bevfusion.yaml',recursive=True)
        cls.cfg=Config(recursive_eval(c))

    def make_model(self):
        dc=deepcopy(self.cfg.model.heads.object)
        dc.in_channels=8;dc.share_conv_channel=4
        dc.bbox_coder.max_num=3
        dc.bbox_coder.pc_range=[-4,-4,-5,4,4,3]
        dc.bbox_coder.voxel_size=[1,1];dc.bbox_coder.out_size_factor=1
        dc.test_cfg.nms_type=['circle']*6
        dc.train_cfg.grid_size=[8,8,8];dc.train_cfg.out_size_factor=1
        dc.train_cfg.voxel_size=[1,1,1];dc.train_cfg.point_cloud_range=[-4,-4,-5,4,4,3]
        dc.train_cfg.max_objs=4
        dc.instance_selection.score_threshold=0.0
        oc=deepcopy(self.cfg.model.heads.occ)
        oc.in_dim=8;oc.out_dim=8;oc.Dz=2
        oc.coordinate_transform.in_dim=8;oc.coordinate_transform.out_dim=8
        oc.coordinate_transform.point_range=[-4,4,-4,4,0,0]
        oc.coordinate_transform.point_num=[8,8,1]
        oc.coordinate_transform.lidar_point_range=[-4,4,-4,4]
        oc.instance_fusion.instance_channels=4;oc.instance_fusion.bev_channels=8
        # 只测试真实任务头接线，避免编码器消耗；forward_heads 使用同一个生产方法。
        model=InstanceBEVFusion.__new__(InstanceBEVFusion);nn.Module.__init__(model)
        model.heads=nn.ModuleDict({'object':build_head(dc),'occ':build_head(oc)})
        model.loss_scale={'object':1.,'occ':1.};model.instance_to_occ=True
        return model

    def test_forward_source_and_selection_parity(self):
        model=self.make_model().eval();head=model.heads['object']
        head.instance_score_threshold=0.1
        bev=torch.randn(2,8,8,8,requires_grad=True)
        pred,source=head.forward_with_features(bev)
        original=head(bev,[{},{}])
        for a,b in zip(pred,original):
            for key in a[0]: torch.testing.assert_close(a[0][key],b[0][key])
        metas=[dict(box_type_3d=LiDARInstance3DBoxes)]*2
        inst=head.build_instances(pred,source,metas)
        baseline=head.get_bboxes(pred,metas)
        flat=source.permute(0,2,3,1).reshape(2,-1,4)
        for b,item in enumerate(inst):
            torch.testing.assert_close(item['features'],flat[b,item['source_indices']])
            boxes,scores,labels=baseline[b];order=scores.argsort(descending=True)
            torch.testing.assert_close(item['boxes'][:,:3],boxes.gravity_center[order])
            torch.testing.assert_close(item['boxes'][:,3:],boxes.tensor[order,3:])
            torch.testing.assert_close(item['scores'],scores[order])
            self.assertFalse(item['boxes'].requires_grad)
            self.assertTrue(item['features'].requires_grad)

    def test_shared_postprocessing_parity(self):
        #* (11.16) 两样本、不同注入阈值及空候选：检测结果不应受注入阈值影响。
        from unittest.mock import patch
        from mmdet3d.models.heads.bbox import proposal_head, centerpoint
        model = self.make_model().eval()
        head = model.heads['object']
        pred, source = head.forward_with_features(torch.randn(2, 8, 8, 8))
        metas = [dict(box_type_3d=LiDARInstance3DBoxes)] * 2
        # rotate 算子需要 CUDA；CPU 对比用同一确定性桩验证外围筛选/索引，不声称验证 CUDA NMS。
        def cpu_nms(boxes, scores, *args, **kwargs):
            return scores.argsort(descending=True)[:kwargs['post_max_size']]
        for nms_type in ['circle', 'rotate']:
            head.test_cfg.nms_type = [nms_type] * 6
            with patch.object(proposal_head, 'nms_gpu', cpu_nms), patch.object(centerpoint, 'nms_gpu', cpu_nms):
                expected = head.get_bboxes(pred, metas)
                for threshold in [0., .7, 1.]:
                    head.instance_score_threshold = threshold
                    with patch.object(head, '_decode_task', wraps=head._decode_task) as decode:
                        inst, actual = head.build_instances(pred, source, metas, return_detections=True)
                    self.assertEqual(decode.call_count, 6)
                    old_inst = head.build_instances(pred, source, metas)
                    for b in range(2):
                        torch.testing.assert_close(actual[b][0].tensor, expected[b][0].tensor)
                        torch.testing.assert_close(actual[b][1], expected[b][1])
                        torch.testing.assert_close(actual[b][2], expected[b][2])
                        for key in inst[b]:
                            torch.testing.assert_close(inst[b][key], old_inst[b][key])
                empty_pred = tuple([{k: v.clone() for k, v in t[0].items()}] for t in pred)
                for t in empty_pred:
                    t[0]['heatmap'].fill_(-100)
                inst, det = head.build_instances(empty_pred, source, metas, return_detections=True)
                self.assertTrue(all(len(x['boxes']) == 0 for x in inst))
                self.assertTrue(all(len(x[0]) == 0 for x in det))

    def test_occupancy_loss_reaches_detection_features(self):
        model=self.make_model().train();head=model.heads['object']
        bev=torch.randn(2,8,8,8,requires_grad=True)
        predictions,source=head.forward_with_features(bev)
        # 固定几何候选，让此检查仅关注来源 gather 的梯度，不依赖随机预测质量。
        detached=tuple([{k:v.detach().clone() for k,v in task[0].items()}] for task in predictions)
        for task in detached:
            task[0]['heatmap'].fill_(-20)
        detached[0][0]['heatmap'][:,0,3,3]=8
        detached[0][0]['reg'].zero_()
        detached[0][0]['height'].zero_()
        detached[0][0]['dim'].fill_(1.38629436)
        detached[0][0]['rot'][:,0].zero_();detached[0][0]['rot'][:,1].fill_(1)
        metas=[dict(box_type_3d=LiDARInstance3DBoxes)]*2
        instances=head.build_instances(detached,source,metas)
        self.assertTrue(all(len(item['boxes']) == 1 for item in instances))
        matrix=torch.eye(4)[None].repeat(2,1,1)
        # detach 共享BEV直达占据的路径，确保检测 shared_conv 梯度只能来自实例注入。
        occ=model.heads['occ'](bev.detach(),matrix,matrix,matrix,instances=instances)
        occ.square().mean().backward()
        grad=head.shared_conv.conv.weight.grad
        self.assertIsNotNone(grad);self.assertGreater(grad.abs().sum().item(),0)

    def test_optional_gt_quality_filter(self):
        #* (11.23) 验证可选GT质量筛选及同类别约束，默认路径由其余回归测试覆盖。
        from unittest.mock import patch
        model = self.make_model().train()
        head = model.heads['object']
        head.instance_train_filter = 'gt_quality'
        item = dict(
            boxes=torch.tensor([[0., 0., 0., 2., 2., 2., 0., 0., 0.],
                                [10., 0., 0., 2., 2., 2., 0., 0., 0.]]),
            scores=torch.tensor([.9, .9]), labels=torch.tensor([0, 0]),
            source_indices=torch.tensor([0, 1]), task_ids=torch.tensor([0, 0]))
        # GT tensor使用底面中心z=-1，对应重心中心z=0。
        gt = LiDARInstance3DBoxes(torch.tensor([[0., 0., -1., 2., 2., 2., 0., 0., 0.]]), box_dim=9)
        with patch.object(LiDARInstance3DBoxes, 'overlaps', return_value=torch.zeros(2, 1)):
            kept = head._filter_instances_by_gt(
                item, dict(box_type_3d=LiDARInstance3DBoxes), gt, torch.tensor([0]))
        self.assertEqual(kept['source_indices'].tolist(), [0])

        # 同位置但类别不同也不能借用该GT通过几何筛选。
        wrong = {key: value[:1].clone() for key, value in item.items()}
        wrong['labels'].fill_(1)
        with patch.object(LiDARInstance3DBoxes, 'overlaps', return_value=torch.zeros(1, 1)):
            kept = head._filter_instances_by_gt(
                wrong, dict(box_type_3d=LiDARInstance3DBoxes), gt, torch.tensor([0]))
        self.assertEqual(len(kept['boxes']), 0)

    def test_train_eval_and_disabled_parity(self):
        model=self.make_model();head=model.heads['object']
        bev=torch.randn(2,8,8,8,requires_grad=True)
        matrix=torch.eye(4)[None].repeat(2,1,1)
        metas=[dict(box_type_3d=LiDARInstance3DBoxes)]*2
        gt=[LiDARInstance3DBoxes(torch.tensor([[0.,0.,0.,2.,2.,1.,0.,0.,0.]]),box_dim=9),
            LiDARInstance3DBoxes(torch.empty(0,9),box_dim=9)]
        labels=[torch.tensor([0]),torch.empty(0,dtype=torch.long)]
        heatmaps,anno,inds,masks=head.get_targets(gt,labels)
        targets=dict(heatmaps=heatmaps,anno_boxes=anno,inds=inds,masks=masks)
        model.train()
        losses=model.forward_heads(bev,metas,matrix,matrix,matrix,gt,labels,
                                   object_targets=targets,voxel_semantics=torch.zeros(2,8,8,2,dtype=torch.long),
                                   mask_camera=torch.ones(2,8,8,2,dtype=torch.bool))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(bev.grad).all())
        self.assertTrue(all(p.grad is not None for p in model.heads['occ'].instance_fusion.parameters()))
        model.eval()
        from unittest.mock import patch
        with torch.no_grad(), patch.object(head, 'get_bboxes', side_effect=AssertionError('重复后处理')):
            outputs=model.forward_heads(bev,metas,matrix,matrix,matrix)
        self.assertEqual(len(outputs),2);self.assertEqual(outputs[0]['occ_pred'].shape,(8,8,2))
        # None 调用父类路径，空候选应给出相同结果。
        occ=model.heads['occ']
        empty=[dict(features=bev.new_empty(0,4),boxes=bev.new_empty(0,9),scores=bev.new_empty(0),
                    labels=torch.empty(0,dtype=torch.long)) for _ in range(2)]
        torch.testing.assert_close(occ(bev,matrix,matrix,matrix),occ(bev,matrix,matrix,matrix,instances=empty))


if __name__=='__main__':
    unittest.main()
