"""Run with python -m unittest discover -s tests/test_models -p test_bev_guided_refine.py."""
import unittest
from types import SimpleNamespace
import torch
from mmdet3d.models.fusers.bev_history_fusion import BEVHistoryFusion
from mmdet3d.models.fusion_models.streaming_bevfusion import StreamingBEVFusion


class TestGuidedRefine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.base = BEVHistoryFusion(32, 16).eval()
        self.guided = BEVHistoryFusion(32, 16, mode='guided_refine', guide_channels=16).eval()
        result = self.guided.load_state_dict(self.base.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(all(k.startswith(('refine_net.guide.', 'refine_net.channel_bias.',
                                         'refine_net.spatial_bias.')) for k in result.missing_keys))
        self.current = torch.randn(2, 32, 8, 10, requires_grad=True)
        self.history = torch.randn_like(self.current)
        self.obs = torch.randn_like(self.current)
        self.dt = torch.tensor([.5, -1.])
        self.mask = torch.ones(2, 8, 10, dtype=torch.bool)
        self.mask[1] = False
        self.mask[0, :, :2] = False

    def test_zero_guidance_equivalence_train_and_eval(self):
        for training in (False, True):
            self.base.train(training); self.guided.train(training)
            expected = self.base(self.current, self.history)
            actual = self.guided(self.current, self.history, self.obs, self.dt, self.mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gradients_and_no_history(self):
        out = self.guided(self.current, self.history, self.obs, self.dt, self.mask)
        out.square().sum().backward()
        for p in self.guided.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(self.guided.refine_net.channel_bias.weight.grad.abs().sum().item(), 0)
        self.guided.zero_grad()
        with torch.no_grad():
            self.guided.refine_net.channel_bias.bias.fill_(2)
            self.guided.refine_net.spatial_bias.bias.fill_(-1)
        mask = torch.zeros_like(self.mask)
        actual = self.guided(self.current, self.history, self.obs, self.dt, mask)
        torch.testing.assert_close(actual, self.base(self.current, self.history), rtol=0, atol=0)
        actual.sum().backward()
        for name, p in self.guided.refine_net.named_parameters():
            if name.startswith(('guide.', 'channel_bias.', 'spatial_bias.')):
                self.assertIsNotNone(p.grad)
                self.assertEqual(p.grad.count_nonzero().item(), 0)

    def test_streaming_fallback_and_warp_border(self):
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 10), indexing='ij')
        grid = torch.stack((grid_x, grid_y), -1)[None].repeat(2, 1, 1, 1)
        hist = self.history.clone(); hist[:, :, :, :2] = 0
        common = dict(training=True, _stream_metas=[{'timestamp': .5}, {'timestamp': 1.}],
                      bev_cache=SimpleNamespace(observation=self.obs, metas=[{'timestamp': 0.}, {'timestamp': 0.}]),
                      _history_alignment={'grid': grid, 'valid_mask': self.mask})
        a = SimpleNamespace(history_fusion=self.base, **common)
        b = SimpleNamespace(history_fusion=self.guided, **common)
        for history, valid in [(None, None), (hist, torch.tensor([True, False]))]:
            x = StreamingBEVFusion.process_history(a, self.current, history, valid)
            y = StreamingBEVFusion.process_history(b, self.current, history, valid)
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_modes_and_parameter_counts(self):
        self.assertEqual(BEVHistoryFusion(32,16,use_refine=False).mode, 'identity')
        self.assertEqual(BEVHistoryFusion(32,16,use_controlled_update=True).mode, 'controlled_update')
        self.assertEqual(BEVHistoryFusion(32,16,use_controlled_update=True,mode='refine').mode, 'refine')
        for mode, count in [('refine',313698), ('controlled_update',640866), ('guided_refine',449955)]:
            model = BEVHistoryFusion(512,128,mode=mode)
            self.assertEqual(sum(p.numel() for p in model.history_updater.parameters()),count)
            model.load_state_dict(model.state_dict(), strict=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast_backward(self):
        model = self.guided.cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(self.current.cuda(), self.history.cuda(), self.obs.cuda(), self.dt.cuda(), self.mask.cuda())
        out.float().square().mean().backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
