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

#include <c10/core/Device.h>
#include <torch/torch.h>

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

c10::Device get_device(int device_id) {
  c10::Device device(torch::kCPU);
  if (device_id >= 0) {
    device = c10::Device(torch::kCUDA, device_id);
  }
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
