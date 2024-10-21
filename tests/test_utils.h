/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

// Generate random vector data for testing
template <typename T>
std::vector<T> generate_random(const std::vector<int64_t>& shape) {

  int64_t num_values = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  std::vector<T> data(num_values);

  std::mt19937 generator;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);
  std::uniform_real_distribution<T> dist((T)0, (T)1);

  auto r = [&]() {
    return dist(generator);
  };

  std::generate(data.begin(), data.end(), r);

  return data;

}

// Generate random names to use as model keys to avoid conflicts between tests
std::string generate_random_name(int length) {

  const std::string character_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  std::mt19937 generator;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);
  std::uniform_int_distribution<> dist(0, character_set.size() - 1);

  std::string name;
  for (int i = 0; i < length; ++i) {
    name += character_set[dist(generator)];
  }
  return name;
}

// Get raw data pointer from vector. If dev is GPU, this routine will allocate GPU memory and copy.
template <typename T>
T* get_data_ptr(std::vector<T>& data, int dev) {
  T* data_ptr;
#ifdef ENABLE_GPU
  if (dev == TORCHFORT_DEVICE_CPU) {
    data_ptr = data.data();
  } else {
    CHECK_CUDA(cudaMalloc(&data_ptr, data.size() * sizeof(T(0))));
    CHECK_CUDA(cudaMemcpy(data_ptr, data.data(), data.size() * sizeof(T(0)), cudaMemcpyHostToDevice));
  }
#else
  data_ptr = data.data();
#endif

  return data_ptr;
}

// Free raw data pointer. If dev is GPU, this routine will free GPU memory.
template <typename T>
void free_data_ptr(T* data_ptr, int dev) {
#ifdef ENABLE_GPU
  if (dev != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaFree(data_ptr));
  }
#endif
}

// Routines to copy vector data to and from GPU.
#ifdef ENABLE_GPU
template <typename T>
void copy_to_host_vector(std::vector<T>& data, T* data_ptr) {
    CHECK_CUDA(cudaMemcpy(data.data(), data_ptr, data.size()*sizeof(T(0)), cudaMemcpyDeviceToHost));
}
template <typename T>
void copy_from_host_vector(T* data_ptr, std::vector<T>& data) {
    CHECK_CUDA(cudaMemcpy(data_ptr, data.data(), data.size()*sizeof(T(0)), cudaMemcpyHostToDevice));
}
#endif

