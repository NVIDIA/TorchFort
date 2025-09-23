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

#include <torch/torch.h>

namespace torchfort {

// Simple struct to store miscellaneous model state (e.g. iteration count)
struct ModelState {
  int64_t step_train;
  int64_t step_inference;
  int64_t step_train_current; // training step of current run (ignoring restarted state)
  torch::Device device = torch::Device(torch::kCPU);

  // General option settings
  int32_t report_frequency;
  bool enable_wandb_hook;
  bool verbose;
  std::filesystem::path report_file;

  void save(const std::string& fname);
  void load(const std::string& fname);
};

} // namespace torchfort
