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

#include <memory>
#include <vector>

#include <torch/enum.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "internal/base_loss.h"
#include "internal/defines.h"
#include "internal/param_map.h"

namespace torchfort {

struct L1Loss : BaseLoss {
  void setup(const ParamMap& params) override;

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& labels,
                        const std::vector<torch::Tensor>& extra_args) override;

  torch::nn::L1Loss module;
};

struct MSELoss : BaseLoss {
  void setup(const ParamMap& params) override;

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& labels,
                        const std::vector<torch::Tensor>& extra_args) override;

  torch::nn::MSELoss module;
};

struct TorchscriptLoss : BaseLoss {
  void setup(const ParamMap& params) override;

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& labels,
                        const std::vector<torch::Tensor>& extra_args) override;

  std::shared_ptr<torch::jit::Module> module_jit;
};

// Creating loss_registry.
BEGIN_LOSS_REGISTRY

// Add entries for new losses in this section. First argument to REGISTER_LOSS is
// a string key and the second argument is the class name.
REGISTER_LOSS(L1, L1Loss)
REGISTER_LOSS(MSE, MSELoss)
REGISTER_LOSS(torchscript, TorchscriptLoss)

END_LOSS_REGISTRY

} // namespace torchfort
