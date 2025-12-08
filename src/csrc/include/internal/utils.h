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

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <c10/core/TensorOptions.h>
#ifdef ENABLE_GPU
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include "internal/exceptions.h"
#include "internal/nvtx.h"

namespace torchfort {

// Function to convert string to lowercase and remove whitespace
std::string sanitize(std::string s);

// Function to convert a string to a filename
std::string filename_sanitize(std::string s);

// Function to return torch device from integer device value
torch::Device get_device(int device);

// Function to return torch device from pointer
torch::Device get_device(const void* ptr);

template <typename T> torch::Dtype make_type() {
  if (std::is_same<T, float>::value) {
    return torch::kFloat32;
  } else if (std::is_same<T, int32_t>::value) {
    return torch::kInt32;
  } else if (std::is_same<T, int64_t>::value) {
    return torch::kInt64;
  } else if (std::is_same<T, double>::value) {
    return torch::kFloat64;
  } else {
    THROW_INVALID_USAGE("datatype not implemented");
  }
}

enum MemoryLayout { RowMajor = 0, ColMajor = 1 };

template <MemoryLayout L, typename T> torch::Tensor get_tensor(T* tensor_ptr, size_t dim, int64_t* shape) {
  torchfort::nvtx::rangePush("get_tensor");
  // Set tensor options
  auto dev = get_device(tensor_ptr);
  torch::TensorOptions options = torch::TensorOptions().device(dev);

  // Get type
  auto type = make_type<T>();
  options = options.dtype(type);

  // Create shape
  std::vector<int64_t> sizes(dim);
  switch (L) {
  case RowMajor:
    for (size_t i = 0; i < dim; ++i) {
      sizes[i] = shape[i];
    }
    break;
  case ColMajor:
    // For column major input data, reverse the shape order
    for (size_t i = 0; i < dim; ++i) {
      sizes[i] = shape[dim - i - 1];
    }
    break;
  }
  torch::IntArrayRef size = c10::makeArrayRef<int64_t>(sizes);

  // Create tensor
  auto tensor = torch::from_blob(
      tensor_ptr, sizes, [](void* ptr) {}, options);
  torchfort::nvtx::rangePop();
  return tensor;
}

// Helper function to convert string reduction names to torch enums.
template <typename T> T get_torch_reduction(const std::string& s) {
  if (s == "mean") {
    return torch::kMean;
  } else if (s == "sum") {
    return torch::kSum;
  } else if (s == "none") {
    return torch::kNone;
  } else {
    THROW_INVALID_USAGE("Unknown reduction type encountered.");
  }
}

// Helper function for printing tensor shapes
std::string print_tensor_shape(torch::Tensor tensor);

// Helper function to get the lrs
std::vector<double> get_current_lrs(const char* name);

#ifdef ENABLE_GPU
// Helper function to set the device and stream with device checks
void set_device_and_stream(c10::cuda::OptionalCUDAStreamGuard& stream_guard, c10::cuda::OptionalCUDAGuard& cuda_guard,
                           torch::Device device, cudaStream_t ext_stream);
#endif
} // namespace torchfort
