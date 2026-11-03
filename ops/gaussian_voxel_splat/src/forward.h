#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "config.h"

namespace FORWARD {
void preprocess(const int P, const int* means_int, const int* radii,
                const dim3 grid, uint32_t* tiles_touched);
void render(const int N, const float* pts, const int* points_int,
            const dim3 grid, const uint2* ranges,
            const uint32_t* gaussian_list, const float* means,
            const float* precision, const float* opacity,
            const float* features, const float sigma_limit_sq,
            const bool normalized, float* out, float* weight_sum);
}
