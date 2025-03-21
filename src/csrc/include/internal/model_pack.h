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
