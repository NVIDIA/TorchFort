/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <c10/core/TensorOptions.h>
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

} // namespace torchfort
