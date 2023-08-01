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

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "internal/base_loss.h"
#include "internal/base_lr_scheduler.h"
#include "internal/base_model.h"
#include "internal/lr_schedulers.h"
#include "internal/model_state.h"
#include "internal/model_wrapper.h"

namespace torchfort {

void check_params(const std::set<std::string>& supported_params, const std::set<std::string>& provided_params);

ParamMap get_params(const YAML::Node& params_node);

std::shared_ptr<ModelWrapper> get_model(const YAML::Node& model_node);

std::shared_ptr<BaseLoss> get_loss(const YAML::Node& loss_node);

std::shared_ptr<torch::optim::Optimizer> get_optimizer(const YAML::Node& optimizer_node,
                                                       std::vector<torch::Tensor> parameters);

std::shared_ptr<torch::optim::Optimizer> get_optimizer(const YAML::Node& optimizer_node,
                                                       const std::shared_ptr<ModelWrapper>& model);

std::shared_ptr<BaseLRScheduler> get_lr_scheduler(const YAML::Node& lr_scheduler_node,
                                                  const std::shared_ptr<torch::optim::Optimizer>& optimizer);

std::shared_ptr<ModelState> get_state(const char* name, const YAML::Node& state_node);
} // namespace torchfort
