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

#include "internal/base_loss.h"
#include "internal/base_lr_scheduler.h"
#include "internal/distributed.h"
#include "internal/model_state.h"
#include "internal/model_wrapper.h"

namespace torchfort {

// Simple struct to group model, optimizer, lr scheduler, state, and comm objects
struct ModelPack {
  std::shared_ptr<ModelWrapper> model;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
  std::shared_ptr<BaseLRScheduler> lr_scheduler;
  std::shared_ptr<BaseLoss> loss;
  std::shared_ptr<Comm> comm;
  std::shared_ptr<ModelState> state;
  int grad_accumulation_steps = 1;
};

void save_model_pack(const ModelPack& model_pack, const std::string& fname, bool save_optimizer = true);
void load_model_pack(ModelPack& model_pack, const std::string& fname, bool load_optimizer = true);

} // namespace torchfort
