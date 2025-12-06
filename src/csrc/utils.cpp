/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#ifdef ENABLE_GPU
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include "internal/defines.h"
#include "internal/model_pack.h"
#include "internal/utils.h"

namespace torchfort {

// Declaration of external global variables
extern std::unordered_map<std::string, ModelPack> models;

std::string sanitize(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::string filename_sanitize(std::string s) {
  // remove trailing whitespace
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());

  // replace intermediate whitespace
  s = std::regex_replace(s, std::regex(" "), "_");

  // replace all / with _:
  s = std::regex_replace(s, std::regex("/"), "-");

  return s;
}

torch::Device get_device(int device) {
  torch::Device device_torch(torch::kCPU);
  if (device != TORCHFORT_DEVICE_CPU) {
#ifdef ENABLE_GPU
    device_torch = torch::Device(torch::kCUDA, device);
#else
    THROW_NOT_SUPPORTED(
        "Attempted to place a model or other component on GPU but TorchFort was build without GPU support.");
#endif
  }
  return device_torch;
}

torch::Device get_device(const void* ptr) {
  torch::Device device = torch::Device(torch::kCPU);
#ifdef ENABLE_GPU
  cudaPointerAttributes attr;
  CHECK_CUDA(cudaPointerGetAttributes(&attr, ptr));
  switch (attr.type) {
  case cudaMemoryTypeHost:
  case cudaMemoryTypeUnregistered:
    device = torch::Device(torch::kCPU);
    break;
  case cudaMemoryTypeManaged:
  case cudaMemoryTypeDevice:
    device = torch::Device(torch::kCUDA);
    break;
  }
#endif
  return device;
}

std::string print_tensor_shape(torch::Tensor tensor) {
  std::string shapestr = "(";
  for (int i = 0; i < tensor.dim(); ++i)
    shapestr += std::to_string(tensor.size(i)) + ",";
  shapestr.pop_back();
  shapestr += ")";
  return shapestr;
}

std::vector<double> get_current_lrs(const char* name) {
  auto optimizer = models[name].optimizer;
  std::vector<double> learnings_rates(optimizer->param_groups().size());
  if (learnings_rates.size() > 0) {
    for (const auto i : c10::irange(optimizer->param_groups().size())) {
      learnings_rates[i] = optimizer->param_groups()[i].options().get_lr();
    }
  }
  return learnings_rates;
}

#ifdef ENABLE_GPU
int getStreamDevice(cudaStream_t stream) {
  CUdevice device;

#if CUDART_VERSION >= 12080
  if (IS_CUDA_DRV_FUNC_AVAILABLE(cuStreamGetDevice)) {
    CHECK_CUDA_DRV(cuStreamGetDevice((CUstream)stream, &device));
    return (int)device;
  }
#endif

  CUcontext streamCtx, savedCtx;
  CHECK_CUDA_DRV(cuCtxGetCurrent(&savedCtx));
  CHECK_CUDA_DRV(cuStreamGetCtx((CUstream)stream, &streamCtx));
  CHECK_CUDA_DRV(cuCtxSetCurrent(streamCtx));
  CHECK_CUDA_DRV(cuCtxGetDevice(&device));
  CHECK_CUDA_DRV(cuCtxSetCurrent(savedCtx));
  return (int)device;
}

void set_device_and_stream(c10::cuda::OptionalCUDAStreamGuard& stream_guard, c10::cuda::OptionalCUDAGuard& cuda_guard, torch::Device device, cudaStream_t ext_stream) {
  if (device.is_cuda()) {
    cuda_guard.set_device(device);
    if (ext_stream) {
      int ext_stream_device;
      ext_stream_device = getStreamDevice(ext_stream);
      if (ext_stream_device != device.index()) {
        std::stringstream ss;
        ss << "The provided external stream is on device " << get_device(ext_stream_device) << " but the model is on device " << device << ".";
        THROW_INVALID_USAGE(ss.str());
      }
      stream_guard.reset_stream(c10::cuda::getStreamFromExternal(ext_stream, device.index()));
    }
  }
}
#endif

} // namespace torchfort
