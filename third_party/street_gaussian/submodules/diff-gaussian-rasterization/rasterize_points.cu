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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
	const torch::Tensor& semantics,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int S = semantics.size(1);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_semantic = torch::full({S, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M, S,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(),
		semantics.contiguous().data<float>(),
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_alpha.contiguous().data<float>(),
		out_semantic.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, out_alpha, out_semantic, radii, geomBuffer, binningBuffer, imgBuffer);
}

__device__ void insertTopContribution(
    const int K,
    const int gaussian_id,
    const float alpha,
    const float transmittance,
    const float weight,
    const float depth,
    const int depth_order,
    int* out_ids,
    float* out_alpha,
    float* out_T,
    float* out_weight,
    float* out_depth,
    int* out_depth_order)
{
    int pos = K;
    for (int k = 0; k < K; ++k)
    {
        if (weight > out_weight[k])
        {
            pos = k;
            break;
        }
    }
    if (pos == K)
        return;
    for (int k = K - 1; k > pos; --k)
    {
        out_ids[k] = out_ids[k - 1];
        out_alpha[k] = out_alpha[k - 1];
        out_T[k] = out_T[k - 1];
        out_weight[k] = out_weight[k - 1];
        out_depth[k] = out_depth[k - 1];
        out_depth_order[k] = out_depth_order[k - 1];
    }
    out_ids[pos] = gaussian_id;
    out_alpha[pos] = alpha;
    out_T[pos] = transmittance;
    out_weight[pos] = weight;
    out_depth[pos] = depth;
    out_depth_order[pos] = depth_order;
}

__global__ void querySelectedPixelContributorsCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int* __restrict__ selected_pixels,
    const int N,
    const int K,
    const int W,
    const int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ depths,
    const float4* __restrict__ conic_opacity,
    int* __restrict__ out_ids,
    float* __restrict__ out_alpha,
    float* __restrict__ out_T,
    float* __restrict__ out_weight,
    float* __restrict__ out_depth,
    int* __restrict__ out_depth_order)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    int x = selected_pixels[idx * 2 + 0];
    int y = selected_pixels[idx * 2 + 1];
    int base = idx * K;
    for (int k = 0; k < K; ++k)
    {
        out_ids[base + k] = -1;
        out_alpha[base + k] = 0.0f;
        out_T[base + k] = 0.0f;
        out_weight[base + k] = 0.0f;
        out_depth[base + k] = 0.0f;
        out_depth_order[base + k] = -1;
    }
    if (x < 0 || x >= W || y < 0 || y >= H)
        return;

    const int horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const int tile_x = x / BLOCK_X;
    const int tile_y = y / BLOCK_Y;
    const uint2 range = ranges[tile_y * horizontal_blocks + tile_x];
    const float2 pixf = { (float)x, (float)y };

    float T = 1.0f;
    int contributor = 0;
    for (uint32_t range_idx = range.x; range_idx < range.y; ++range_idx)
    {
        int gaussian_id = point_list[range_idx];
        contributor++;
        float2 xy = points_xy_image[gaussian_id];
        float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        float4 con_o = conic_opacity[gaussian_id];
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;
        float alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;
        float test_T = T * (1.0f - alpha);
        if (test_T < 0.0001f)
            break;
        float weight = alpha * T;
        insertTopContribution(
            K,
            gaussian_id,
            alpha,
            T,
            weight,
            depths[gaussian_id],
            contributor,
            out_ids + base,
            out_alpha + base,
            out_T + base,
            out_weight + base,
            out_depth + base,
            out_depth_order + base);
        T = test_T;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansContribCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
	const torch::Tensor& semantics,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
    const torch::Tensor& selected_pixels,
    const int top_k)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  if (selected_pixels.ndimension() != 2 || selected_pixels.size(1) != 2) {
    AT_ERROR("selected_pixels must have dimensions (num_pixels, 2)");
  }
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int S = semantics.size(1);
  const int N = selected_pixels.size(0);
  const int K = top_k;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  torch::Tensor out_ids = torch::full({N, K}, -1, int_opts);
  torch::Tensor out_alpha = torch::zeros({N, K}, float_opts);
  torch::Tensor out_T = torch::zeros({N, K}, float_opts);
  torch::Tensor out_weight = torch::zeros({N, K}, float_opts);
  torch::Tensor out_depth = torch::zeros({N, K}, float_opts);
  torch::Tensor out_depth_order = torch::full({N, K}, -1, int_opts);
  if (P == 0 || N == 0 || K <= 0) {
    return std::make_tuple(out_ids, out_alpha, out_T, out_weight, out_depth, out_depth_order);
  }

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_render_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha_image = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_semantic = torch::full({S, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, int_opts);
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int M = 0;
  if(sh.size(0) != 0) {
    M = sh.size(1);
  }
  int rendered = CudaRasterizer::Rasterizer::forward(
    geomFunc,
    binningFunc,
    imgFunc,
    P, degree, M, S,
    background.contiguous().data<float>(),
    W, H,
    means3D.contiguous().data<float>(),
    sh.contiguous().data_ptr<float>(),
    colors.contiguous().data<float>(),
    semantics.contiguous().data<float>(),
    opacity.contiguous().data<float>(),
    scales.contiguous().data_ptr<float>(),
    scale_modifier,
    rotations.contiguous().data_ptr<float>(),
    cov3D_precomp.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    campos.contiguous().data<float>(),
    tan_fovx,
    tan_fovy,
    prefiltered,
    out_color.contiguous().data<float>(),
    out_render_depth.contiguous().data<float>(),
    out_alpha_image.contiguous().data<float>(),
    out_semantic.contiguous().data<float>(),
    radii.contiguous().data<int>(),
    debug);

  if (rendered <= 0) {
    return std::make_tuple(out_ids, out_alpha, out_T, out_weight, out_depth, out_depth_order);
  }

  char* geom_chunk = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
  char* binning_chunk = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
  char* img_chunk = reinterpret_cast<char*>(imgBuffer.contiguous().data_ptr());
  CudaRasterizer::GeometryState geomState = CudaRasterizer::GeometryState::fromChunk(geom_chunk, P);
  CudaRasterizer::BinningState binningState = CudaRasterizer::BinningState::fromChunk(binning_chunk, rendered);
  CudaRasterizer::ImageState imgState = CudaRasterizer::ImageState::fromChunk(img_chunk, W * H);

  const int threads = 128;
  const int blocks = (N + threads - 1) / threads;
  querySelectedPixelContributorsCUDA<<<blocks, threads>>>(
    imgState.ranges,
    binningState.point_list,
    selected_pixels.contiguous().data_ptr<int>(),
    N,
    K,
    W,
    H,
    geomState.means2D,
    geomState.depths,
    geomState.conic_opacity,
    out_ids.contiguous().data_ptr<int>(),
    out_alpha.contiguous().data_ptr<float>(),
    out_T.contiguous().data_ptr<float>(),
    out_weight.contiguous().data_ptr<float>(),
    out_depth.contiguous().data_ptr<float>(),
    out_depth_order.contiguous().data_ptr<int>());

  return std::make_tuple(out_ids, out_alpha, out_T, out_weight, out_depth, out_depth_order);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& dL_dout_semantic,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& alphas,
	const torch::Tensor& semantics,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  const int S = dL_dout_semantic.size(0);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dsemantic = torch::zeros({P, S}, means3D.options());

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, S,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  semantics.contiguous().data<float>(),
	  alphas.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depth.contiguous().data<float>(),
	  dL_dout_alpha.contiguous().data<float>(),
	  dL_dout_semantic.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  dL_dsemantic.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dsemantic);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<torch::Tensor, torch::Tensor>
 RasterizeGaussiansfilterCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor means2D = torch::full({P, 2}, 0, means3D.options());
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  

  if(P != 0)
  {
	  int M = 0;

	  CudaRasterizer::Rasterizer::visible_filter(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, M,
		W, H,
		means3D.contiguous().data<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		radii.contiguous().data<int>(),
		means2D.contiguous().data<float>(),
		debug);
  }
  return std::make_tuple(radii, means2D);
}
