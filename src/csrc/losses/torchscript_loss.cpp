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

#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/enum.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/losses.h"
#include "internal/param_map.h"
#include "internal/setup.h"
#include "internal/utils.h"

namespace torchfort {

void TorchscriptLoss::setup(const ParamMap& params) {
  std::string jit_loss_fname;
  try {
    jit_loss_fname = params.get_param<std::string>("filename")[0];
  } catch (std::out_of_range) {
    THROW_INVALID_USAGE("filename parameter is required for torchscript loss type.");
  }

  if (!std::filesystem::exists(jit_loss_fname)) {
    THROW_INVALID_USAGE(jit_loss_fname + " does not exist.");
  }

  module_jit = std::shared_ptr<torch::jit::Module>(new torch::jit::Module);
  *module_jit = torch::jit::load(jit_loss_fname);
}

torch::Tensor TorchscriptLoss::forward(const std::vector<torch::Tensor>& inputs,
                                       const std::vector<torch::Tensor>& labels,
                                       const std::vector<torch::Tensor>& extra_args) {
  std::vector<torch::jit::IValue> inputs_jit;
  inputs_jit.insert(inputs_jit.end(), inputs.begin(), inputs.end());
  inputs_jit.insert(inputs_jit.end(), labels.begin(), labels.end());
  inputs_jit.insert(inputs_jit.end(), extra_args.begin(), extra_args.end());

  auto result = module_jit->forward(inputs_jit);
  if (!result.isTensor()) {
    THROW_INVALID_USAGE("TorchscriptLoss only supports returning a single loss tensor.");
  }
  return result.toTensor();
}

} // namespace torchfort
