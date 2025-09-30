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

#include <vector>

#include <torch/torch.h>

#include "internal/models.h"
#include "internal/param_map.h"
#include "internal/setup.h"

namespace torchfort {

// MLP model in C++ using libtorch
void MLPModel::setup(const ParamMap& params) {
  // Extract params from input map.
  std::set<std::string> supported_params{"dropout", "flatten_non_batch_dims", "layer_sizes"};
  check_params(supported_params, params.keys());

  dropout = params.get_param<double>("dropout", 0.0)[0];
  flatten_non_batch_dims = params.get_param<bool>("flatten_non_batch_dims", true)[0];
  layer_sizes = params.get_param<int>("layer_sizes");

  // Construct and register submodules.
  for (int i = 0; i < layer_sizes.size() - 1; ++i) {
    fc_layers.push_back(
        register_module("fc" + std::to_string(i), torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1])));
    if (i < layer_sizes.size() - 2) {
      biases.push_back(register_parameter("b" + std::to_string(i), torch::zeros(layer_sizes[i + 1])));
    }
  }
}

// Implement the forward function.
std::vector<torch::Tensor> MLPModel::forward(const std::vector<torch::Tensor>& inputs) {
  if (inputs.size() > 1)
    THROW_INVALID_USAGE("Built-in MLP model does not support multiple input tensors.");

  auto x = inputs[0];

  if (flatten_non_batch_dims) {
    x = x.reshape({x.size(0), -1});
  }

  for (int i = 0; i < layer_sizes.size() - 1; ++i) {
    if (i < layer_sizes.size() - 2) {
      x = torch::relu(fc_layers[i]->forward(x) + biases[i]);
      x = torch::dropout(x, dropout, is_training());
    } else {
      x = fc_layers[i]->forward(x);
    }
  }
  return std::vector<torch::Tensor>{x};
}

} // namespace torchfort
