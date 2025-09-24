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

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/enum.h>
#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/losses.h"
#include "internal/param_map.h"
#include "internal/setup.h"
#include "internal/utils.h"

namespace torchfort {

void MSELoss::setup(const ParamMap& params) {
  std::set<std::string> supported_params{"reduction"};
  check_params(supported_params, params.keys());

  auto options = torch::nn::MSELossOptions();
  try {
    std::string reduction = params.get_param<std::string>("reduction")[0];
    options = options.reduction(get_torch_reduction<torch::nn::MSELossOptions::reduction_t>(reduction));
  } catch (std::out_of_range) {
    // use default
  }

  module = torch::nn::MSELoss(options);
}

torch::Tensor MSELoss::forward(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& labels,
                               const std::vector<torch::Tensor>& extra_args) {
  if (inputs.size() != 1 || labels.size() != 1 || extra_args.size() != 0) {
    THROW_INVALID_USAGE("MSELoss only supports one input tensor, one label tensor, and no extra arguments.");
  }
  auto x = inputs[0];
  auto y = labels[0];
  return module(x.flatten(), y.flatten());
}

} // namespace torchfort
