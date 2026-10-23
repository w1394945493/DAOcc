"""直接按顺序检查几何、概率和梯度；--cuda额外核对复制的CUDA内核。"""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.GaussDAOcc.geometry import GaussianPredictor, inverse_covariance
from models.GaussDAOcc.aggregation import GaussianAggregation


def _run_checks():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--full-model', action='store_true', help='构建完整DAOcc并用真实数据检查前向')
    parser.add_argument('--build-only', action='store_true', help='配合--full-model，只在CPU检查模型构建')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/gauss_daocc/gauss_daocc.yaml'))
    parser.add_argument('--daocc-root', help='已编译mmdet3d扩展的DAOcc路径')
    parser.add_argument('--ckpt', help='可选：原DAOcc或已训练高斯模型权重；不传则随机初始化')
    parser.add_argument('--sample-index', type=int, default=0, help='验证集起始索引')
    parser.add_argument('--num-samples', type=int, default=1, help='连续检查多少帧，每帧batch=1')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.build_only and not args.full_model:
        parser.error('--build-only必须和--full-model一起使用')
    if args.sample_index < 0 or args.num_samples < 1 or args.device < 0:
        parser.error('sample-index/device不能为负，num-samples必须大于0')
    torch.manual_seed(3)
    bounds, shape = [-1., -1.5, -.5, 1., 1.5, .5], [4, 6, 2]
    #* 1. 使用非方形网格，检查XYZ顺序以及一批两个样本的前向、反向。
    predictor = GaussianPredictor(8, bounds, [2, 3, 2], scale_init=.35)
    aggregate = GaussianAggregation(bounds, shape, backend='torch_reference')
    bev = torch.randn(2, 8, 6, 4, requires_grad=True)
    gaussians = predictor(bev)
    output = aggregate(gaussians)
    assert output.shape == (2, 4, 6, 2, 18)
    torch.testing.assert_close(output.exp().sum(-1), torch.ones_like(output[..., 0]))
    loss = -output[..., 3].mean() - output[..., 17].mean()
    loss.backward()
    assert torch.isfinite(bev.grad).all() and bev.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
    assert (gaussians['means'] > predictor.lower).all() and (gaussians['means'] < predictor.upper).all()
    print('通过：非方形网格、batch=2、概率归一化、BEV及高斯预测头梯度。')

    #* 2. 旋转后的逆协方差应正定；单位旋转下与1/s²一致。
    scales = torch.tensor([[.2, .4, .6]])
    q = torch.tensor([[1., 0., 0., 0.]])
    torch.testing.assert_close(inverse_covariance(scales, q), torch.diag_embed(scales.reciprocal().square()))
    rotated = inverse_covariance(scales, torch.tensor([[1., .2, -.3, .4]]))
    assert (torch.linalg.eigvalsh(rotated) > 0).all()
    print('通过：协方差构造及旋转后正定性。')

    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('本机无可用CUDA，不能执行CUDA数值检查')
        #* 3. 不只验证能运行：用不同尺度及非单位旋转，核对CUDA前向和各属性梯度。
        # 避开高斯中心与查询点精确重合的退化情况，防止上游乘积求导的0/0近似。
        means = torch.tensor([[[-.24, -.32, -.12], [.37, .41, .16]]], device='cuda')
        s = torch.tensor([[[.31, .42, .28], [.27, .36, .33]]], device='cuda')
        q = torch.tensor([[[1., .1, .2, -.15], [1., -.2, .15, .1]]], device='cuda')
        semantic = torch.softmax(torch.randn(1, 2, 17, device='cuda'), -1)
        g = dict(means=means, scales=s, opacity=torch.tensor([[.4, .7]], device='cuda'),
                 semantics=torch.nn.functional.pad(semantic, (0, 1)), inverse_covariance=inverse_covariance(s, q))
        g1 = {k: v.detach().clone().requires_grad_(k != 'scales') for k, v in g.items()}
        g2 = {k: v.detach().clone().requires_grad_(k != 'scales') for k, v in g.items()}
        reference = GaussianAggregation(bounds, shape, backend='torch_reference').cuda()(g1)
        actual = GaussianAggregation(bounds, shape, backend='cuda').cuda()(g2)
        #* 同时打印概率与log概率误差：接近零的概率取log后会放大浮点误差。
        # 保留原断言阈值，不通过放宽阈值掩盖CUDA或参考公式的问题。
        probability_error = (actual.exp()-reference.exp()).abs().max().item()
        log_error = (actual-reference).abs().max().item()
        print('CUDA前向最大误差：概率={:.8g}，log概率={:.8g}'.format(
            probability_error, log_error), flush=True)
        torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
        weights = torch.randn_like(actual)
        (reference*weights).sum().backward()
        (actual*weights).sum().backward()
        # 对称协方差六项打包：CUDA把非对角导数累加到上三角，比较对称化梯度。
        for key in g1:
            if key == 'scales':
                continue
            a, b = g1[key].grad, g2[key].grad
            if key == 'inverse_covariance':
                a, b = a+a.transpose(-1, -2), b+b.transpose(-1, -2)
            torch.testing.assert_close(a, b, atol=5e-3, rtol=5e-3)
        print('通过：CUDA与参考实现的前向及反向数值比较。')

    if args.full_model:
        from scripts.check_gauss_daocc_full import check_full_model
        check_full_model(args)


def main():
    #* 数值核对必须使用一致的FP32精度。PyTorch 1.10默认允许TF32，
    # einsum/@可能走TF32矩阵乘法，而自定义CUDA标量内核仍按FP32计算。
    # 关闭AMP并不等于关闭TF32，因此这里需要单独设置。
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print('数值检查：已关闭TF32，保持原前向/反向误差阈值。', flush=True)
        _run_checks()
    finally:
        # 只影响此次检查；即使断言失败，也恢复调用者原有设置。
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


if __name__ == '__main__':
    main()
