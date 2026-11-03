"""小网格前向/反向数值对照，不依赖DAOcc数据集。"""
import torch
from ops.gaussian_voxel_splat import gaussian_voxel_splat


def reference(centers, scales, rotations, opacities, features, pc_range, grid_size,
              sigma_limit, accumulation):
    sx, sy, sz = grid_size
    lo = centers.new_tensor(pc_range[:3])
    hi = centers.new_tensor(pc_range[3:])
    step = (hi - lo) / centers.new_tensor(grid_size)
    ix, iy, iz = torch.meshgrid(
        torch.arange(sx, device=centers.device),
        torch.arange(sy, device=centers.device),
        torch.arange(sz, device=centers.device), indexing="ij")
    pts = lo + (torch.stack((ix, iy, iz), -1).reshape(-1, 3) + 0.5) * step
    delta = pts[None] - centers[:, None]
    local = torch.einsum("gni,gij->gnj", delta, rotations)
    d2 = ((local / scales[:, None].clamp_min(1e-4)) ** 2).sum(-1)
    weights = opacities.reshape(-1, 1) * torch.exp(-0.5 * d2)
    weights = weights * (d2 <= sigma_limit ** 2)
    value = weights.T @ features
    if accumulation == "normalized":
        value = value / weights.sum(0).clamp_min(1e-6)[:, None]
    return value.reshape(sx, sy, sz, 128).permute(3, 2, 1, 0)[None]


def clone_leaf(value):
    return value.detach().clone().requires_grad_(True)


def main():
    torch.manual_seed(7)
    device = "cuda"
    count = 5
    base = {
        "centers": torch.rand(count, 3, device=device) * 3.0 + 0.5,
        "scales": torch.rand(count, 3, device=device) * 0.35 + 0.45,
        "rotations": torch.eye(3, device=device)[None].repeat(count, 1, 1),
        "opacities": torch.rand(count, 1, device=device) * 0.7 + 0.2,
        "features": torch.randn(count, 128, device=device) * 0.2,
    }
    pc_range = [0., 0., 0., 4., 4., 4.]
    grid_size = [8, 7, 6]
    sigma = 2.5
    for mode in ("normalized", "sum"):
        cuda_inputs = {key: clone_leaf(value) for key, value in base.items()}
        ref_inputs = {key: clone_leaf(value) for key, value in base.items()}
        actual, _ = gaussian_voxel_splat(**cuda_inputs, pc_range=pc_range,
            grid_size=grid_size, sigma_limit=sigma, accumulation=mode)
        expected = reference(**ref_inputs, pc_range=pc_range,
            grid_size=grid_size, sigma_limit=sigma, accumulation=mode)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        print(mode, "forward_max", float((actual - expected).abs().max()),
              "forward_rel_l2", float(torch.linalg.vector_norm(actual - expected)
              / torch.linalg.vector_norm(expected).clamp_min(1e-8)))
        for key in base:
            ga, ge = cuda_inputs[key].grad, ref_inputs[key].grad
            # 旋转矩阵受R^T R=I约束，只比较合法无穷小旋转方向上的梯度。
            # 原始9元素梯度包含离开SO(3)流形的分量，不具有可比性。
            if key == "rotations":
                ga = ga - ga.transpose(-1, -2)
                ge = ge - ge.transpose(-1, -2)
            print(" ", key, "grad_max", float((ga-ge).abs().max()),
                  "grad_rel_l2", float(torch.linalg.vector_norm(ga-ge)
                  / torch.linalg.vector_norm(ge).clamp_min(1e-8)))


if __name__ == "__main__":
    main()
