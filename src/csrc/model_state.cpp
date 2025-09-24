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

#include <string>

#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/model_state.h"

namespace torchfort {

void ModelState::save(const std::string& fname) {
  torch::serialize::OutputArchive archive;
  archive.write("step_train", torch::IValue(step_train));
  archive.write("step_inference", torch::IValue(step_inference));
  archive.write("device", torch::IValue(device));
  archive.save_to(fname);
}

void ModelState::load(const std::string& fname) {
  if (!std::filesystem::exists(fname)) {
    THROW_INVALID_USAGE(fname + " does not exist.");
  }

  torch::serialize::InputArchive archive;
  archive.load_from(fname);

  torch::IValue ivalue;
  if (!archive.try_read("step_train", ivalue)) {
    THROW_INVALID_USAGE(fname + " is missing required data.");
  }
  step_train = ivalue.to<int64_t>();

  if (!archive.try_read("step_inference", ivalue)) {
    THROW_INVALID_USAGE(fname + " is missing required data.");
  }
  step_inference = ivalue.to<int64_t>();

  if (!archive.try_read("device", ivalue)) {
    THROW_INVALID_USAGE(fname + " is missing required data.");
  }
  device = ivalue.to<torch::Device>();
}

} // namespace torchfort
