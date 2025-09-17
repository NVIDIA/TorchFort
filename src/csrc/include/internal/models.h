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

#include "internal/base_model.h"
#include "internal/defines.h"
#include "internal/param_map.h"

namespace torchfort {

// MLP model in C++ using libtorch
struct MLPModel : BaseModel, public std::enable_shared_from_this<BaseModel> {
  void setup(const ParamMap& params) override;
  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;

  double dropout;
  bool flatten_non_batch_dims;
  std::vector<int> layer_sizes;

  // Use one of many "standard library" modules.
  std::vector<torch::nn::Linear> fc_layers;
  std::vector<torch::Tensor> biases;
};

struct CriticMLPModel : BaseModel, public std::enable_shared_from_this<BaseModel> {
  void setup(const ParamMap& params) override;
  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;

  double dropout;
  std::vector<int> layer_sizes;

  // Use one of many "standard library" modules.
  std::vector<torch::nn::Linear> fc_layers;
  std::vector<torch::Tensor> biases;
};

struct SACMLPModel : BaseModel, public std::enable_shared_from_this<BaseModel> {
  void setup(const ParamMap& params) override;
  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;

  double dropout;
  bool flatten_non_batch_dims;
  std::vector<int> layer_sizes;
  bool state_dependent_sigma;

  // A SAC Model has a common encoder and two output layers for mu and log-sigma
  std::vector<torch::nn::Linear> encoder_layers;
  std::vector<torch::nn::Linear> out_layers;
  std::vector<torch::Tensor> biases;
  std::vector<torch::Tensor> out_biases;
};

struct ActorCriticMLPModel : BaseModel, public std::enable_shared_from_this<BaseModel> {
  void setup(const ParamMap& params) override;
  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;

  double dropout;
  std::vector<int> encoder_layer_sizes, actor_layer_sizes, value_layer_sizes;
  bool state_dependent_sigma;

  // An AC Model has a common encoder and then an MLP for actor and one for value
  std::vector<torch::nn::Linear> encoder_layers, actor_layers, value_layers;
  std::vector<torch::Tensor> encoder_biases, actor_biases, value_biases;
};

// Creating model_registry.
BEGIN_MODEL_REGISTRY

// Add entries for new models in this section.
REGISTER_MODEL(MLP, MLPModel)
REGISTER_MODEL(CriticMLP, CriticMLPModel)
REGISTER_MODEL(SACMLP, SACMLPModel)
REGISTER_MODEL(ActorCriticMLP, ActorCriticMLPModel)

END_MODEL_REGISTRY

} // namespace torchfort
