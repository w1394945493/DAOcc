#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "config.h"

namespace BACKWARD {
void preprocess(const int N, const int* points_int, const dim3 grid, int* voxel2pts);
void render(const int P, const uint32_t* offsets,
            const uint32_t* voxel_keys_unsorted, const int* voxel2pts,
            const float* pts, const float* means, const float* precision,
            const float* opacity, const float* features, const float* out,
            const float* weight_sum, const float* out_grad,
            const float sigma_limit_sq, const bool normalized,
            float* means_grad, float* opacity_grad, float* features_grad,
            float* precision_grad);
}
