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

#include "internal/rl/setup.h"

#include "internal/exceptions.h"
#include "internal/setup.h"

namespace torchfort {

namespace rl {

std::shared_ptr<ReplayBuffer> get_replay_buffer(const YAML::Node& rb_node, float gamma, int nstep,
                                                RewardReductionMode nstep_reward_reduction, int rb_device) {
  if (!rb_node["type"]) {
    THROW_INVALID_USAGE("Missing type field in replay_buffer section in configuration file.");
  }
  std::string rb_type = sanitize(rb_node["type"].as<std::string>());

  if (!rb_node["parameters"]) {
    THROW_INVALID_USAGE("Missing parameters section in replay_buffer section in configuration file.");
  }
  auto params = get_params(rb_node["parameters"]);

  std::set<std::string> supported_params{"type", "max_size", "min_size", "n_envs",
                                         "alpha", "beta0", "beta_max", "beta_steps"};
  check_params(supported_params, params.keys());

  auto max_size = static_cast<size_t>(params.get_param<int>("max_size")[0]);
  auto min_size = static_cast<size_t>(params.get_param<int>("min_size")[0]);
  auto n_envs   = static_cast<size_t>(params.get_param<int>("n_envs", 1)[0]);

  if (rb_type == "uniform") {
    return std::make_shared<UniformReplayBuffer>(max_size, min_size, n_envs, gamma, nstep,
                                                 nstep_reward_reduction, rb_device);
  } else if (rb_type == "prioritized") {
    float  alpha      = params.get_param<float>("alpha",    0.6f)[0];
    float  beta0      = params.get_param<float>("beta0",    0.4f)[0];
    float  beta_max   = params.get_param<float>("beta_max", 1.0f)[0];
    size_t beta_steps = static_cast<size_t>(params.get_param<int>("beta_steps", 100000)[0]);
    return std::make_shared<PrioritizedReplayBuffer>(max_size, min_size, n_envs, gamma, nstep,
                                                     nstep_reward_reduction, alpha, beta0, beta_max,
                                                     beta_steps, rb_device);
  } else {
    THROW_INVALID_USAGE("Unknown replay_buffer type: " + rb_type);
  }
}

} // namespace rl

} // namespace torchfort
