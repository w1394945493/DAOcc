#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void preprocessCUDA(const int P, const int* means_int, const int* radii,
                               const dim3 grid, uint32_t* tiles_touched) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;
    uint3 rect_min, rect_max;
    getRect(means_int + 3 * idx, radii + 3 * idx, rect_min, rect_max, grid);
    tiles_touched[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y)
                       * (rect_max.z - rect_min.z);
}

template <uint32_t CHANNELS>
__global__ void renderCUDA(
    const int N, const float* __restrict__ pts, const int* __restrict__ points_int,
    const dim3 grid, const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ gaussian_list, const float* __restrict__ means,
    const float* __restrict__ precision, const float* __restrict__ opacity,
    const float* __restrict__ features, const float sigma_limit_sq,
    const bool normalized, float* __restrict__ out,
    float* __restrict__ weight_sum) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N) return;
    const int* pi = points_int + 3 * idx;
    const int voxel_idx = pi[0] * grid.y * grid.z + pi[1] * grid.z + pi[2];
    const float3 p = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};
    const uint2 range = ranges[voxel_idx];
    float accum[CHANNELS] = {0.0f};
    float total_weight = 0.0f;
    for (uint32_t k = range.x; k < range.y; ++k) {
        const int g = gaussian_list[k];
        const float dx = means[3 * g] - p.x;
        const float dy = means[3 * g + 1] - p.y;
        const float dz = means[3 * g + 2] - p.z;
        const float* q = precision + 6 * g;
        const float d2 = q[0] * dx * dx + q[1] * dy * dy + q[2] * dz * dz
                       + 2.0f * (q[3] * dx * dy + q[4] * dy * dz + q[5] * dx * dz);
        if (d2 > sigma_limit_sq) continue;
        const float kernel = expf(-0.5f * d2);
        const float weight = opacity[g] * kernel;
        total_weight += weight;
        for (int ch = 0; ch < CHANNELS; ++ch)
            accum[ch] += features[g * CHANNELS + ch] * weight;
    }
    weight_sum[idx] = total_weight;
    const float denom = fmaxf(total_weight, 1e-6f);
    for (int ch = 0; ch < CHANNELS; ++ch)
        out[idx * CHANNELS + ch] = normalized ? accum[ch] / denom : accum[ch];
}

void FORWARD::preprocess(const int P, const int* means_int, const int* radii,
                         const dim3 grid, uint32_t* tiles_touched) {
    preprocessCUDA<<<(P + 255) / 256, 256>>>(P, means_int, radii, grid, tiles_touched);
}

void FORWARD::render(
    const int N, const float* pts, const int* points_int, const dim3 grid,
    const uint2* ranges, const uint32_t* gaussian_list, const float* means,
    const float* precision, const float* opacity, const float* features,
    const float sigma_limit_sq, const bool normalized, float* out,
    float* weight_sum) {
    renderCUDA<NUM_CHANNELS><<<(N + 255) / 256, 256>>>(
        N, pts, points_int, grid, ranges, gaussian_list, means, precision,
        opacity, features, sigma_limit_sq, normalized, out, weight_sum);
}
