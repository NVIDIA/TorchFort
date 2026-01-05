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
template <typename T> std::vector<T> generate_random(const std::vector<int64_t>& shape) {

  int64_t num_values = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  std::vector<T> data(num_values);

  std::mt19937 generator;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);
  std::uniform_real_distribution<T> dist((T)0, (T)1);

  auto r = [&]() { return dist(generator); };

  std::generate(data.begin(), data.end(), r);

  return data;
}

// Generate constant vector data for testing
template <typename T> std::vector<T> generate_constant(const std::vector<int64_t>& shape, T value) {

  int64_t num_values = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  std::vector<T> data(num_values, value);

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
template <typename T> T* get_data_ptr(std::vector<T>& data, int dev) {
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
template <typename T> void free_data_ptr(T* data_ptr, int dev) {
#ifdef ENABLE_GPU
  if (dev != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaFree(data_ptr));
  }
#endif
}

// Routines to copy vector data to and from GPU.
#ifdef ENABLE_GPU
template <typename T> void copy_to_host_vector(std::vector<T>& data, T* data_ptr) {
  CHECK_CUDA(cudaMemcpy(data.data(), data_ptr, data.size() * sizeof(T(0)), cudaMemcpyDeviceToHost));
}
template <typename T> void copy_from_host_vector(T* data_ptr, std::vector<T>& data) {
  CHECK_CUDA(cudaMemcpy(data_ptr, data.data(), data.size() * sizeof(T(0)), cudaMemcpyHostToDevice));
}
#endif

bool check_current_device(int expected_device) {
#ifdef ENABLE_GPU
  int device;
  CHECK_CUDA(cudaGetDevice(&device));

  return expected_device == TORCHFORT_DEVICE_CPU || device == expected_device;
#endif
  return true;
}
