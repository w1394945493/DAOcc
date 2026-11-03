#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianVoxelSplatCUDA(const torch::Tensor& pts, const torch::Tensor& points_int,
    const torch::Tensor& means, const torch::Tensor& means_int,
    const torch::Tensor& opacity, const torch::Tensor& features,
    const torch::Tensor& radii, const torch::Tensor& precision,
    int H, int W, int D, float sigma_limit, bool normalized);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianVoxelSplatBackwardCUDA(const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer, const torch::Tensor& image_buffer,
    int H, int W, int D, int rendered, const torch::Tensor& means,
    const torch::Tensor& pts, const torch::Tensor& points_int,
    const torch::Tensor& precision, const torch::Tensor& opacity,
    const torch::Tensor& features, const torch::Tensor& out,
    const torch::Tensor& weight_sum, const torch::Tensor& out_grad,
    float sigma_limit, bool normalized);
