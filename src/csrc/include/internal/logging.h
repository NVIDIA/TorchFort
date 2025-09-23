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

#include <filesystem>
#include <string>

#include "internal/model_pack.h"

namespace torchfort {

namespace logging {

enum level { info, warn, error, wandb };

std::string log_level_prefix(level log_level);
void write(const std::filesystem::path& filename, const std::string& message, level log_level);
void print(const std::string& message, level log_level);

} // namespace logging

// Declaration of external global variables
extern std::unordered_map<std::string, ModelPack> models;

// specialized logging routines
template <typename T>
void wandb_log(std::shared_ptr<ModelState> state, std::shared_ptr<Comm> comm, const char* name, const char* metric_name,
               int64_t step, T value) {
  if (state->enable_wandb_hook) {
    std::stringstream os;
    os << "model: " << name << ", ";
    os << "step: " << step << ", ";
    os << metric_name << ": " << value;
    if (!comm || (comm && comm->rank == 0)) {
      torchfort::logging::write(state->report_file, os.str(), torchfort::logging::wandb);
    }
  }
}

template <typename T> void wandb_log(const char* name, const char* metric_name, int64_t step, T value) {
  auto state = models[name].state.get();
  if (state->enable_wandb_hook) {
    std::stringstream os;
    os << "model: " << name << ", ";
    os << "step: " << step << ", ";
    os << metric_name << ": " << value;
    if (!models[name].comm || (models[name].comm && models[name].comm->rank == 0)) {
      torchfort::logging::write(state->report_file, os.str(), torchfort::logging::wandb);
    }
  }
}

} // namespace torchfort
