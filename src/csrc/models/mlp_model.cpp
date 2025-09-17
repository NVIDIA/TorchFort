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

#include <vector>

#include <torch/torch.h>

#include "internal/models.h"
#include "internal/param_map.h"
#include "internal/setup.h"

namespace torchfort {

// MLP model in C++ using libtorch
void MLPModel::setup(const ParamMap& params) {
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
std::vector<torch::Tensor> MLPModel::forward(const std::vector<torch::Tensor>& inputs) {
  // concatenate inputs
  auto x = torch::cat(inputs, 1);
  x = x.reshape({x.size(0), -1});

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
