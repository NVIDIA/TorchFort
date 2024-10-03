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

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs,
                        const std::vector<torch::Tensor>& labels,
                        const std::vector<torch::Tensor>& extra_args) override;

  torch::nn::L1Loss module;
};

struct MSELoss : BaseLoss {
  void setup(const ParamMap& params) override;

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs,
                        const std::vector<torch::Tensor>& labels,
                        const std::vector<torch::Tensor>& extra_args) override;

  torch::nn::MSELoss module;
};

struct TorchscriptLoss : BaseLoss {
  void setup(const ParamMap& params) override;

  torch::Tensor forward(const std::vector<torch::Tensor>& inputs,
                        const std::vector<torch::Tensor>& labels,
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
