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

#include <vector>

#include <torch/torch.h>

#include "internal/utils.h"

namespace torchfort {
struct TensorList {
  template <MemoryLayout L, typename T> void add_tensor(T* data, size_t dim, int64_t* shape) {
    auto tensor = get_tensor<L>(data, dim, shape);
    tensors.push_back(tensor);
    tensors_original_.push_back(tensor);
  };

  void to(torch::Device device, bool non_blocking = false) {
    for (auto& t : tensors) {
      t = t.to(device, non_blocking);
    }
  };

  void reset() { tensors = tensors_original_; }

  std::vector<torch::Tensor> tensors;
  // To preserve references to external data, we store the original tensor objects
  std::vector<torch::Tensor> tensors_original_;
};
} // namespace torchfort
