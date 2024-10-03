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

torch::Tensor MSELoss::forward(const std::vector<torch::Tensor>& inputs,
                               const std::vector<torch::Tensor>& labels,
                               const std::vector<torch::Tensor>& extra_args) {
  if (inputs.size() != 1 || labels.size() != 1 || extra_args.size() != 0) {
    THROW_INVALID_USAGE("MSELoss only supports one input tensor, one label tensor, and no extra arguments.");
  }
  auto x = inputs[0];
  auto y = labels[0];
  return module(x.flatten(), y.flatten());
}

} // namespace torchfort
