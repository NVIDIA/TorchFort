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
