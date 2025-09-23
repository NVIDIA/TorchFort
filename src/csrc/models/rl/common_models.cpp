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
void CriticMLPModel::setup(const ParamMap& params) {
  // Extract params from input map.
  std::set<std::string> supported_params{"dropout", "layer_sizes"};
  check_params(supported_params, params.keys());

  dropout = params.get_param<double>("dropout", 0.0)[0];
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
std::vector<torch::Tensor> CriticMLPModel::forward(const std::vector<torch::Tensor>& inputs) {

  // makse sure that exactly two tensors are fed, state and action:
  if (inputs.size() != 2) {
    THROW_INVALID_USAGE("You have to provide exactly two tensors (state, action) to the CriticMLPModel");
  }

  // unpack
  auto state = inputs[0];
  auto action = inputs[1];

  // expand dims if necessary
  if (state.dim() == 1) {
    state = state.unsqueeze(0);
  }
  if (action.dim() == 1) {
    action = action.unsqueeze(0);
  }

  // flatten everything beyond dim 0:
  state = state.reshape({state.size(0), -1});
  action = action.reshape({action.size(0), -1});

  // concatenate inputs along feature dimension
  auto x = torch::cat({state, action}, 1);

  // forward pass
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
