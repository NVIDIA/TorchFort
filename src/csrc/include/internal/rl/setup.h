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

#include <yaml-cpp/yaml.h>

#include "internal/rl/replay_buffer.h"

namespace torchfort {

namespace rl {

// Construct a replay buffer from a YAML replay_buffer node.
// The node must contain a "type" field ("uniform" or "prioritized") and a "parameters" sub-node.
// gamma, nstep and nstep_reward_reduction are taken from the enclosing algorithm configuration
// and forwarded to the buffer constructor.
std::shared_ptr<ReplayBuffer> get_replay_buffer(const YAML::Node& rb_node, float gamma, int nstep,
                                                RewardReductionMode nstep_reward_reduction, int rb_device);

} // namespace rl

} // namespace torchfort
