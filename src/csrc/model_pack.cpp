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

#include <filesystem>

#include "internal/defines.h"
#include "internal/model_pack.h"
#include "internal/model_wrapper.h"
#include "internal/utils.h"

namespace torchfort {

void save_model_pack(const ModelPack& model_pack, const std::string& dir, bool save_optimizer) {
  std::filesystem::path root_dir(dir);

  if (!std::filesystem::exists(root_dir)) {
    bool rv = std::filesystem::create_directory(root_dir);
    if (!rv) {
      THROW_INVALID_USAGE("Could not create directory " + root_dir.native() + ".");
    }
  }

  model_pack.state->device = model_pack.model->device();

  auto model_path = root_dir / "model.pt";
  model_pack.model->save(model_path.native());

  if (save_optimizer) {
    auto optimizer_path = root_dir / "optimizer.pt";
    if (!model_pack.optimizer) {
      THROW_INVALID_USAGE("Cannot save checkpoint. Missing optimizer.");
    }
    torch::save(*(model_pack.optimizer), optimizer_path.native());

    auto lr_path = root_dir / "lr.pt";
    if (model_pack.lr_scheduler) {
      model_pack.lr_scheduler->save(lr_path.native());
    }
  }

  auto state_path = root_dir / "state.pt";
  model_pack.state->save(state_path.native());
}

void load_model_pack(ModelPack& model_pack, const std::string& dir, bool load_optimizer) {
  std::filesystem::path root_dir(dir);

  auto state_path = root_dir / "state.pt";
  if (!std::filesystem::exists(state_path)) {
    THROW_INVALID_USAGE("Could not find " + state_path.native() + ".");
  }
  model_pack.state->load(state_path.native());

  auto model_path = root_dir / "model.pt";
  if (!std::filesystem::exists(model_path)) {
    THROW_INVALID_USAGE("Could not find " + model_path.native() + ".");
  }
  model_pack.model->load(model_path.native());

  // Assign optimizer to parameters of loaded model:
  // we need to check if the optimizer is initialized before doing so
  // (some RL models do not have an optimizer attached to them):
  if (model_pack.optimizer) {
    model_pack.optimizer->parameters() = model_pack.model->parameters();
  }

  if (load_optimizer) {
    auto optimizer_path = root_dir / "optimizer.pt";
    if (!std::filesystem::exists(optimizer_path)) {
      THROW_INVALID_USAGE("Could not find " + optimizer_path.native() + ".");
    }
    torch::load(*(model_pack.optimizer), optimizer_path.native(), model_pack.model->device());

    auto lr_path = root_dir / "lr.pt";
    if (std::filesystem::exists(lr_path)) {
      model_pack.lr_scheduler->load(lr_path.native(), *(model_pack.optimizer));
    } else {
      // No LR in checkpoint, disable LR scheduler
      model_pack.lr_scheduler = nullptr;
    }
  }
}

} // namespace torchfort
