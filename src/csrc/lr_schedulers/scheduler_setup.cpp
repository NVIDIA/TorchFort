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

#include <memory>
#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "internal/base_lr_scheduler.h"
#include "internal/exceptions.h"
#include "internal/lr_schedulers.h"
#include "internal/param_map.h"
#include "internal/setup.h"
#include "internal/utils.h"

namespace torchfort {

std::shared_ptr<BaseLRScheduler> get_lr_scheduler(const YAML::Node& lr_scheduler_node,
                                                  const std::shared_ptr<torch::optim::Optimizer>& optimizer) {

  auto type = sanitize(lr_scheduler_node["type"].as<std::string>());
  std::shared_ptr<BaseLRScheduler> lr_scheduler = nullptr;

  auto params = get_params(lr_scheduler_node["parameters"]);
  if (type == "step") {
    std::set<std::string> supported_params{"step_size", "gamma"};
    check_params(supported_params, params.keys());
    int step_size;
    try {
      step_size = params.get_param<int>("step_size")[0];
    } catch (std::out_of_range) {
      THROW_INVALID_USAGE("step_lr: step_size parameter is required.");
    }

    double gamma;
    try {
      gamma = params.get_param<double>("gamma")[0];
    } catch (std::out_of_range) {
      gamma = 0.1; // default
    }
    lr_scheduler = std::shared_ptr<BaseLRScheduler>(new StepLR(*optimizer, step_size, gamma));
  } else if (type == "cosine_annealing") {
    std::set<std::string> supported_params{"T_max", "eta_min"};
    check_params(supported_params, params.keys());
    int T_max;
    try {
      T_max = params.get_param<int>("T_max")[0];
    } catch (std::out_of_range) {
      THROW_INVALID_USAGE("cosine_annealing_lr: T_max parameter is required.");
    }

    double eta_min;
    try {
      eta_min = params.get_param<double>("eta_min")[0];
    } catch (std::out_of_range) {
      eta_min = 0.; // default
    }
    lr_scheduler = std::shared_ptr<BaseLRScheduler>(new CosineAnnealingLR(*optimizer, T_max, eta_min));
  } else if (type == "polynomial") {
    std::set<std::string> supported_params{"total_iters", "power"};
    check_params(supported_params, params.keys());

    int total_iters;
    try {
      total_iters = params.get_param<int>("total_iters")[0];
    } catch (std::out_of_range) {
      THROW_INVALID_USAGE("polynomial_lr: total_iters parameter is required.");
    }

    double power;
    try {
      power = params.get_param<double>("power")[0];
    } catch (std::out_of_range) {
      power = 1.; // default
    }
    lr_scheduler = std::shared_ptr<BaseLRScheduler>(new PolynomialLR(*optimizer, total_iters, power));
  } else if (type == "multistep") {
    std::set<std::string> supported_params{"gamma", "milestones"};
    check_params(supported_params, params.keys());

    double gamma = 0.1;
    try {
      gamma = params.get_param<double>("gamma")[0];
    } catch (std::out_of_range) {
      gamma = 0.1; // default
    }

    std::vector<int> milestones;
    try {
      milestones = params.get_param<int>("milestones");

      if (std::any_of(milestones.begin(), milestones.end(), [](int x) { return x < 0; })) {
        THROW_INVALID_USAGE("multistep_lr: the milestones must be non-negative");
      }

      if (!std::is_sorted(milestones.begin(), milestones.end())) {
        THROW_INVALID_USAGE("multistep_lr: the milestones must be monotonically increasing");
      }
    } catch (std::out_of_range) {
      THROW_INVALID_USAGE("multistep_lr: milestones parameter is required.");
    }
    lr_scheduler = std::shared_ptr<BaseLRScheduler>(new MultiStepLR(*optimizer, milestones, gamma));
  } else if (type == "linear") {
    std::set<std::string> supported_params{"total_iters", "start_factor", "end_factor"};
    check_params(supported_params, params.keys());

    int total_iters;
    try {
      total_iters = params.get_param<int>("total_iters")[0];
    } catch (std::out_of_range) {
      THROW_INVALID_USAGE("linear_lr: total_iters parameter is required.");
    }

    double start_factor;
    try {
      start_factor = params.get_param<double>("start_factor")[0];
    } catch (std::out_of_range) {
      start_factor = 0.3333333; // default
    }

    double end_factor;
    try {
      end_factor = params.get_param<double>("end_factor")[0];
    } catch (std::out_of_range) {
      end_factor = 1.0; // default
    }

    lr_scheduler = std::shared_ptr<BaseLRScheduler>(new LinearLR(*optimizer, total_iters, start_factor, end_factor));
  } else {
    THROW_INVALID_USAGE("Unknown lr_scheduler type provided.");
  }

  return lr_scheduler;
}

} // namespace torchfort
