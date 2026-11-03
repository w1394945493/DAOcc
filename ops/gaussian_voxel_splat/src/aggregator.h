#pragma once
#include <functional>

namespace LocalAggregator {
class Aggregator {
public:
    static int forward(std::function<char*(size_t)> geometryBuffer,
        std::function<char*(size_t)> binningBuffer,
        std::function<char*(size_t)> imageBuffer,
        const int P, const int N, const float* pts, const int* points_int,
        const float* means, const int* means_int, const float* opacity,
        const float* features, const float* precision, const int* radii,
        const int H, const int W, const int D, const float sigma_limit_sq,
        const bool normalized, float* out, float* weight_sum,
        bool debug=false);
    static void backward(const int P, const int R, const int N,
        const int H, const int W, const int D, char* geom_buffer,
        char* binning_buffer, char* image_buffer, const int* points_int,
        int* voxel2pts, const float* pts, const float* means,
        const float* precision, const float* opacity, const float* features,
        const float* out, const float* weight_sum, const float* out_grad,
        const float sigma_limit_sq, const bool normalized,
        float* means_grad, float* opacity_grad, float* features_grad,
        float* precision_grad, bool debug=false);
};
}
