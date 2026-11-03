#include "backward.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void preprocessCUDA(const int N, const int* points_int,
                               const dim3 grid, int* voxel2pts) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N) return;
    const int* p = points_int + 3 * idx;
    voxel2pts[p[0] * grid.y * grid.z + p[1] * grid.z + p[2]] = idx;
}

template <uint32_t CHANNELS>
__global__ void renderCUDA(
    const int P, const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ voxel_keys_unsorted,
    const int* __restrict__ voxel2pts, const float* __restrict__ pts,
    const float* __restrict__ means, const float* __restrict__ precision,
    const float* __restrict__ opacity, const float* __restrict__ features,
    const float* __restrict__ out, const float* __restrict__ weight_sum,
    const float* __restrict__ out_grad, const float sigma_limit_sq,
    const bool normalized, float* __restrict__ means_grad,
    float* __restrict__ opacity_grad, float* __restrict__ features_grad,
    float* __restrict__ precision_grad) {
    auto g = cg::this_grid().thread_rank();
    if (g >= P) return;
    const uint32_t start = g == 0 ? 0 : offsets[g - 1];
    const uint32_t end = offsets[g];
    const float mx = means[3 * g], my = means[3 * g + 1], mz = means[3 * g + 2];
    const float* q = precision + 6 * g;
    const float opa = opacity[g];
    float grad_mean[3] = {0.0f, 0.0f, 0.0f};
    float grad_q[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float grad_opa = 0.0f;
    float grad_feature[CHANNELS] = {0.0f};

    for (uint32_t i = start; i < end; ++i) {
        const int point_idx = voxel2pts[voxel_keys_unsorted[i]];
        if (point_idx < 0) continue;
        const float dx = mx - pts[3 * point_idx];
        const float dy = my - pts[3 * point_idx + 1];
        const float dz = mz - pts[3 * point_idx + 2];
        const float d2 = q[0] * dx * dx + q[1] * dy * dy + q[2] * dz * dz
                       + 2.0f * (q[3] * dx * dy + q[4] * dy * dz + q[5] * dx * dz);
        if (d2 > sigma_limit_sq) continue;
        const float kernel = expf(-0.5f * d2);
        const float weight = opa * kernel;
        const float total = weight_sum[point_idx];
        const float denom = fmaxf(total, 1e-6f);
        float grad_weight = 0.0f;
        for (int ch = 0; ch < CHANNELS; ++ch) {
            const float go = out_grad[point_idx * CHANNELS + ch];
            const float feat = features[g * CHANNELS + ch];
            if (normalized) {
                grad_feature[ch] += go * weight / denom;
                grad_weight += go * (total > 1e-6f
                    ? (feat - out[point_idx * CHANNELS + ch]) / denom
                    : feat / denom);
            } else {
                grad_feature[ch] += go * weight;
                grad_weight += go * feat;
            }
        }
        grad_opa += grad_weight * kernel;
        const float grad_kernel = grad_weight * opa;
        const float coeff = grad_kernel * kernel;
        const float qdx = q[0] * dx + q[3] * dy + q[5] * dz;
        const float qdy = q[3] * dx + q[1] * dy + q[4] * dz;
        const float qdz = q[5] * dx + q[4] * dy + q[2] * dz;
        grad_mean[0] -= coeff * qdx;
        grad_mean[1] -= coeff * qdy;
        grad_mean[2] -= coeff * qdz;
        grad_q[0] += -0.5f * coeff * dx * dx;
        grad_q[1] += -0.5f * coeff * dy * dy;
        grad_q[2] += -0.5f * coeff * dz * dz;
        grad_q[3] += -1.0f * coeff * dx * dy;
        grad_q[4] += -1.0f * coeff * dy * dz;
        grad_q[5] += -1.0f * coeff * dx * dz;
    }
    means_grad[3 * g] = grad_mean[0];
    means_grad[3 * g + 1] = grad_mean[1];
    means_grad[3 * g + 2] = grad_mean[2];
    opacity_grad[g] = grad_opa;
    for (int ch = 0; ch < CHANNELS; ++ch)
        features_grad[g * CHANNELS + ch] = grad_feature[ch];
    for (int j = 0; j < 6; ++j) precision_grad[6 * g + j] = grad_q[j];
}

void BACKWARD::preprocess(const int N, const int* points_int,
                          const dim3 grid, int* voxel2pts) {
    preprocessCUDA<<<(N + 255) / 256, 256>>>(N, points_int, grid, voxel2pts);
}

void BACKWARD::render(
    const int P, const uint32_t* offsets, const uint32_t* voxel_keys_unsorted,
    const int* voxel2pts, const float* pts, const float* means,
    const float* precision, const float* opacity, const float* features,
    const float* out, const float* weight_sum, const float* out_grad,
    const float sigma_limit_sq, const bool normalized, float* means_grad,
    float* opacity_grad, float* features_grad, float* precision_grad) {
    renderCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P, offsets, voxel_keys_unsorted, voxel2pts, pts, means, precision,
        opacity, features, out, weight_sum, out_grad, sigma_limit_sq,
        normalized, means_grad, opacity_grad, features_grad, precision_grad);
}
