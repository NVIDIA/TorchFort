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
  std::vector<int> layer_sizes;

  // Use one of many "standard library" modules.
  std::vector<torch::nn::Linear> fc_layers;
  std::vector<torch::Tensor> biases;
};

struct SACMLPModel : BaseModel, public std::enable_shared_from_this<BaseModel> {
  void setup(const ParamMap& params) override;
  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;

  double dropout;
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
REGISTER_MODEL(SACMLP, SACMLPModel)
REGISTER_MODEL(ActorCriticMLP, ActorCriticMLPModel)

END_MODEL_REGISTRY

} // namespace torchfort
