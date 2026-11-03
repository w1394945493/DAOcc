/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "aggregator_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    const int P,
    const int* points_xyz,
    const uint32_t* offsets,
    uint32_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int* radii,
    const dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint3 rect_min, rect_max;

	getRect(points_xyz + 3 * idx, radii + 3 * idx, rect_min, rect_max, grid);
    
    for (int x = rect_min.x; x < rect_max.x; x++)
    {
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int z = rect_min.z; z < rect_max.z; z++)
            {
                uint32_t key = x * grid.y * grid.z + y * grid.z + z;
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(
	int L, 
	uint32_t* point_list_keys, 
	uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint32_t currtile = point_list_keys[idx];
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1];
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


LocalAggregator::GeometryState LocalAggregator::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

LocalAggregator::ImageState LocalAggregator::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

LocalAggregator::BinningState LocalAggregator::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int LocalAggregator::Aggregator::forward(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer,
    const int P, const int N, const float* pts, const int* points_int,
    const float* means, const int* means_int, const float* opacity,
    const float* features, const float* precision, const int* radii,
    const int H, const int W, const int D, const float sigma_limit_sq,
    const bool normalized, float* out, float* weight_sum, bool debug) {
    size_t geom_size = required<GeometryState>(P);
    char* geom_ptr = geometryBuffer(geom_size);
    GeometryState geom = GeometryState::fromChunk(geom_ptr, P);
    size_t image_size = required<ImageState>(H * W * D);
    char* image_ptr = imageBuffer(image_size);
    ImageState image = ImageState::fromChunk(image_ptr, H * W * D);
    const dim3 grid(H, W, D);
    CHECK_CUDA(FORWARD::preprocess(P, means_int, radii, grid, geom.tiles_touched), debug)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geom.scanning_space, geom.scan_size,
        geom.tiles_touched, geom.point_offsets, P), debug)
    int instances = 0;
    CHECK_CUDA(cudaMemcpy(&instances, geom.point_offsets + P - 1,
        sizeof(int), cudaMemcpyDeviceToHost), debug)
    size_t binning_size = required<BinningState>(instances);
    char* binning_ptr = binningBuffer(binning_size);
    BinningState binning = BinningState::fromChunk(binning_ptr, instances);
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(P, means_int, geom.point_offsets,
        binning.point_list_keys_unsorted, binning.point_list_unsorted, radii, grid);
    CHECK_CUDA(, debug)
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(binning.list_sorting_space,
        binning.sorting_size, binning.point_list_keys_unsorted,
        binning.point_list_keys, binning.point_list_unsorted,
        binning.point_list, instances, 0, 32), debug)
    CHECK_CUDA(cudaMemset(image.ranges, 0, H * W * D * sizeof(uint2)), debug)
    if (instances > 0)
        identifyTileRanges<<<(instances + 255) / 256, 256>>>(
            instances, binning.point_list_keys, image.ranges);
    CHECK_CUDA(, debug)
    CHECK_CUDA(FORWARD::render(N, pts, points_int, grid, image.ranges,
        binning.point_list, means, precision, opacity, features,
        sigma_limit_sq, normalized, out, weight_sum), debug)
    return instances;
}

void LocalAggregator::Aggregator::backward(
    const int P, const int R, const int N, const int H, const int W, const int D,
    char* geom_buffer, char* binning_buffer, char* image_buffer,
    const int* points_int, int* voxel2pts, const float* pts,
    const float* means, const float* precision, const float* opacity,
    const float* features, const float* out, const float* weight_sum,
    const float* out_grad, const float sigma_limit_sq, const bool normalized,
    float* means_grad, float* opacity_grad, float* features_grad,
    float* precision_grad, bool debug) {
    GeometryState geom = GeometryState::fromChunk(geom_buffer, P);
    BinningState binning = BinningState::fromChunk(binning_buffer, R);
    ImageState image = ImageState::fromChunk(image_buffer, H * W * D);
    const dim3 grid(H, W, D);
    CHECK_CUDA(BACKWARD::preprocess(N, points_int, grid, voxel2pts), debug)
    CHECK_CUDA(BACKWARD::render(P, geom.point_offsets,
        binning.point_list_keys_unsorted, voxel2pts, pts, means, precision,
        opacity, features, out, weight_sum, out_grad, sigma_limit_sq,
        normalized, means_grad, opacity_grad, features_grad, precision_grad), debug)
}
