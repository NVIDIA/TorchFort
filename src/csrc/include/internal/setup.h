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
