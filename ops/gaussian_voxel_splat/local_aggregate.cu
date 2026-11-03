#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <functional>
#include "local_aggregate.h"
#include "src/aggregator.h"
#include "src/config.h"

std::function<char*(size_t)> resizeFunctional(torch::Tensor& tensor) {
    return [&tensor](size_t size) {
        tensor.resize_({static_cast<long long>(size)});
        return reinterpret_cast<char*>(tensor.contiguous().data_ptr());
    };
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianVoxelSplatCUDA(const torch::Tensor& pts, const torch::Tensor& points_int,
    const torch::Tensor& means, const torch::Tensor& means_int,
    const torch::Tensor& opacity, const torch::Tensor& features,
    const torch::Tensor& radii, const torch::Tensor& precision,
    int H, int W, int D, float sigma_limit, bool normalized) {
    TORCH_CHECK(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(features.size(1) == NUM_CHANNELS, "features must have 128 channels");
    const int P = means.size(0), N = pts.size(0);
    auto out = torch::zeros({N, NUM_CHANNELS}, means.options());
    auto weight_sum = torch::zeros({N}, means.options());
    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(means.device());
    auto geom = torch::empty({0}, byte_options);
    auto binning = torch::empty({0}, byte_options);
    auto image = torch::empty({0}, byte_options);
    int rendered = LocalAggregator::Aggregator::forward(
        resizeFunctional(geom), resizeFunctional(binning), resizeFunctional(image),
        P, N, pts.contiguous().data_ptr<float>(), points_int.contiguous().data_ptr<int>(),
        means.contiguous().data_ptr<float>(), means_int.contiguous().data_ptr<int>(),
        opacity.contiguous().data_ptr<float>(), features.contiguous().data_ptr<float>(),
        precision.contiguous().data_ptr<float>(), radii.contiguous().data_ptr<int>(),
        H, W, D, sigma_limit * sigma_limit, normalized,
        out.data_ptr<float>(), weight_sum.data_ptr<float>());
    return std::make_tuple(rendered, out, weight_sum, geom, binning, image);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianVoxelSplatBackwardCUDA(const torch::Tensor& geom,
    const torch::Tensor& binning, const torch::Tensor& image,
    int H, int W, int D, int rendered, const torch::Tensor& means,
    const torch::Tensor& pts, const torch::Tensor& points_int,
    const torch::Tensor& precision, const torch::Tensor& opacity,
    const torch::Tensor& features, const torch::Tensor& out,
    const torch::Tensor& weight_sum, const torch::Tensor& out_grad,
    float sigma_limit, bool normalized) {
    const int P = means.size(0), N = pts.size(0);
    auto means_grad = torch::zeros_like(means);
    auto opacity_grad = torch::zeros_like(opacity);
    auto features_grad = torch::zeros_like(features);
    auto precision_grad = torch::zeros_like(precision);
    auto voxel2pts = torch::full({H * W * D}, -1,
        means.options().dtype(torch::kInt32));
    LocalAggregator::Aggregator::backward(P, rendered, N, H, W, D,
        reinterpret_cast<char*>(geom.contiguous().data_ptr()),
        reinterpret_cast<char*>(binning.contiguous().data_ptr()),
        reinterpret_cast<char*>(image.contiguous().data_ptr()),
        points_int.contiguous().data_ptr<int>(), voxel2pts.data_ptr<int>(),
        pts.contiguous().data_ptr<float>(), means.contiguous().data_ptr<float>(),
        precision.contiguous().data_ptr<float>(), opacity.contiguous().data_ptr<float>(),
        features.contiguous().data_ptr<float>(), out.contiguous().data_ptr<float>(),
        weight_sum.contiguous().data_ptr<float>(), out_grad.contiguous().data_ptr<float>(),
        sigma_limit * sigma_limit, normalized, means_grad.data_ptr<float>(),
        opacity_grad.data_ptr<float>(), features_grad.data_ptr<float>(),
        precision_grad.data_ptr<float>());
    return std::make_tuple(means_grad, opacity_grad, features_grad, precision_grad);
}
