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
