/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
