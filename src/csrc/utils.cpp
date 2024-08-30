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

#include <algorithm>
#include <regex>
#include <string>

#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/utils.h"

namespace torchfort {

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
#ifdef ENABLE_GPU
  if (device != TORCHFORT_DEVICE_CPU) {
    device_torch = torch::Device(torch::kCUDA, device);
  }
#endif
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
      device = torch::Device(torch::kCPU); break;
    case cudaMemoryTypeManaged:
    case cudaMemoryTypeDevice:
      device = torch::Device(torch::kCUDA); break;
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

} // namespace torchfort
