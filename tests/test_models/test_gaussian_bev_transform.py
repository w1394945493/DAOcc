"""Run from DAOcc: python -m unittest discover -s tests/test_models -p test_gaussian_bev_transform.py -v."""
import unittest
from pathlib import Path
import torch
from torchpack.utils.config import Config as TorchpackConfig
from mmcv import Config as MMCVConfig
from mmdet3d.utils import recursive_eval
from mmdet3d.models.builder import build_vtransform
from mmdet3d.models.vtransforms.gaussian_bev.geometry import GaussianGeometry
from mmdet3d.models.vtransforms.gaussian_bev.renderer import GaussianBEVRenderer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_iter.yaml'
NEW = ROOT / 'configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_gaussian_bev_iter.yaml'


def load(path):
    cfg = TorchpackConfig()
    cfg.load(str(path), recursive=True)
    return recursive_eval(cfg)


class GaussianBEVTest(unittest.TestCase):
    def test_configuration_only_changes_transform_and_fuser(self):
        old, new = load(BASE), load(NEW)
        self.assertEqual(new['model']['encoders']['camera']['vtransform']['type'], 'GaussianBEVTransform')
        self.assertEqual(new['model']['fuser']['in_channels'], [128, 256])
        build_vtransform(new['model']['encoders']['camera']['vtransform'])
        old['model']['encoders']['camera'].pop('vtransform')
        new['model']['encoders']['camera'].pop('vtransform')
        old['model']['fuser'].pop('in_channels')
        new['model']['fuser'].pop('in_channels')
        self.assertEqual(old, new)

    def test_geometry_with_image_and_lidar_augmentation(self):
        geo = GaussianGeometry((4, 6), (2, 3), 2, 1., 5., .5)
        eye = torch.eye(4).view(1,1,4,4)
        intr = eye.clone(); intr[0,0,0,0] = 2.; intr[0,0,1,1] = 3.
        cam = eye.clone()
        cam[0,0,:3,:3] = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        cam[0,0,:3,3] = torch.tensor([1.,2.,3.])
        post = eye.clone()
        post[0,0,0,0] = -2.; post[0,0,1,1] = 2.
        post[0,0,0,3] = 6.; post[0,0,1,3] = 1.
        aug = torch.eye(4).unsqueeze(0)
        aug[0,0,0] = -1.5; aug[0,1,1] = 1.5; aug[0,2,2] = 1.5
        aug[0,:3,3] = torch.tensor([2.,-1.,.5])
        logits = torch.zeros(1,2,2,3,requires_grad=True)
        means, cov = geo(logits, intr, cam, post, aug)
        expected = []
        for v in (0.,4.):
            for u in (0.,3.,6.):
                candidates=[]
                for d in (2.,4.):
                    p=torch.tensor([((u-6.)/-2.)/2.*d, ((v-1.)/2.)/3.*d, d])
                    p=cam[0,0,:3,:3]@p+cam[0,0,:3,3]
                    candidates.append(aug[0,:3,:3]@p+aug[0,:3,3])
                expected.append(torch.stack(candidates))
        expected=torch.stack(expected)
        mu=expected.mean(1)
        delta=expected-mu[:,None]
        sigma=(delta[...,None]@delta[...,None,:]).mean(1)*(.5**2/9.)
        torch.testing.assert_close(means[0],mu)
        torch.testing.assert_close(cov[0],sigma.flatten(-2)[:,[0,1,2,4,5,8]])
        (means.square().sum()+cov.sum()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_grid_centers_and_axis_order(self):
        renderer=GaussianBEVRenderer((-54,54,.6),(-54,54,.6))
        point=torch.tensor([-53.7,-53.1,8.,1.])
        # Column is Y index 1, row is X index 0.
        torch.testing.assert_close((renderer.projection@point)[:2],torch.tensor([1.,0.]),atol=1e-5,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_rasterizer_matches_formula_and_gradients(self):
        # Non-square cells also exercise the scale factors in covariance backward.
        renderer=GaussianBEVRenderer((-4,4,1),(-4,4,.5)).cuda()
        mean=torch.tensor([[[.2,.3,0.]]],device='cuda',requires_grad=True)
        cov=torch.tensor([[[.7,.1,0.,.4,0.,.5]]],device='cuda',requires_grad=True)
        opacity=torch.tensor([[[.6]]],device='cuda',requires_grad=True)
        features=torch.linspace(.1,1.,128,device='cuda').view(1,1,128).requires_grad_()
        actual=renderer(features,mean,cov,opacity)
        # Pixel (row=4,col=8) lies well inside the splat, away from thresholds.
        loss=actual[0,:,4,8].sum()
        grad=torch.autograd.grad(loss,(mean,cov,opacity,features))
        mu=torch.stack((mean[0,0,1]*2+7.5,mean[0,0,0]+3.5))
        sigma=torch.stack((cov[0,0,3]*4+.3,cov[0,0,1]*2,
                           cov[0,0,1]*2,cov[0,0,0]+.3)).view(2,2)
        delta=torch.tensor([8.,4.],device='cuda')-mu
        alpha=opacity[0,0,0]*torch.exp(-.5*delta@torch.linalg.inv(sigma)@delta)
        reference=features.sum()*alpha
        expected_grad=torch.autograd.grad(reference,(mean,cov,opacity,features))
        torch.testing.assert_close(loss,reference,atol=1e-4,rtol=1e-4)
        for a,b in zip(grad,expected_grad):
            torch.testing.assert_close(a,b,atol=2e-3,rtol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_empty_and_opacity_sorting(self):
        renderer=GaussianBEVRenderer((-4,4,1),(-4,4,1)).cuda()
        mean=torch.tensor([[[.5,.5,0.],[.5,.5,0.]]],device='cuda',requires_grad=True)
        cov=torch.tensor([[[1.,0.,0.,1.,0.,1.]]*2],device='cuda',requires_grad=True)
        opacity=torch.tensor([[[.2],[.8]]],device='cuda',requires_grad=True)
        features=torch.ones(1,2,128,device='cuda',requires_grad=True)
        with torch.no_grad(): features[:,1]*=2
        result=renderer(features,mean,cov,opacity)
        self.assertAlmostEqual(result[0,0,4,4].item(),2*.8+1*.2*(1-.8),places=5)
        result.sum().backward()
        for x in (features,mean,cov,opacity): self.assertTrue(torch.isfinite(x.grad).all())
        empty_opacity=torch.zeros_like(opacity,requires_grad=True)
        empty=renderer(features,mean,cov,empty_opacity)
        self.assertEqual(empty.abs().sum().item(),0)
        empty.sum().backward()
        self.assertIsNotNone(empty_opacity.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_configured_transform_forward_backward_amp(self):
        from mmcv.runner import wrap_fp16_model
        model=build_vtransform(load(NEW)['model']['encoders']['camera']['vtransform']).cuda()
        wrap_fp16_model(model)
        x=torch.randn(1,6,256,32,88,device='cuda',dtype=torch.float16,requires_grad=True)
        eye=torch.eye(4,device='cuda').view(1,1,4,4).repeat(1,6,1,1)
        intr=eye.clone()
        intr[:,:,0,0]=intr[:,:,1,1]=500
        intr[:,:,0,2]=352; intr[:,:,1,2]=128
        aug=torch.eye(4,device='cuda').unsqueeze(0)
        with torch.cuda.amp.autocast():
            out=model(x,None,eye,aug,eye,eye,intr,eye,eye,aug,None)
        self.assertEqual(tuple(out.shape),(1,128,180,180))
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for branch in (model.predictor.feats,model.predictor.depth,model.predictor.opacity):
            g=branch[-1].weight.grad
            self.assertIsNotNone(g)
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().sum().item(),0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_existing_bevfusion_camera_to_task_heads(self):
        from mmdet3d.models import build_model
        model=build_model(MMCVConfig(load(NEW)).model).cuda().eval()
        eye=torch.eye(4,device='cuda').view(1,1,4,4).repeat(1,6,1,1)
        intr=eye.clone()
        intr[:,:,0,0]=intr[:,:,1,1]=500
        intr[:,:,0,2]=352; intr[:,:,1,2]=128
        aug=torch.eye(4,device='cuda').unsqueeze(0)
        with torch.no_grad():
            images=torch.randn(1,6,3,256,704,device='cuda')
            camera=model.extract_camera_features(images,None,eye,aug,eye,eye,intr,eye,eye,aug,[{}])
            self.assertEqual(tuple(camera.shape),(1,128,180,180))
            # LiDAR feature stub isolates new camera/fusion integration; sparse backbone unchanged.
            fused=model.fuser([camera,torch.zeros(1,256,180,180,device='cuda')])
            decoded=model.decoder['neck'](model.decoder['backbone'](fused))
            occ=model.heads['occ'](decoded,aug,aug,aug)
            det=model.heads['object'](decoded,[{}])
        self.assertEqual(tuple(fused.shape),(1,512,180,180))
        self.assertEqual(tuple(occ.shape),(1,200,200,16,18))
        self.assertTrue(torch.isfinite(occ).all())
        self.assertEqual(len(det),6)


if __name__ == '__main__': unittest.main()
