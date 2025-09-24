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
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "internal/exceptions.h"

namespace torchfort {

class BaseLRScheduler : public torch::optim::LRScheduler {
public:
  BaseLRScheduler(torch::optim::Optimizer& optimizer) : LRScheduler(optimizer) {}

  // Define generic save/load functionalities. Specialize in derived schedulers if
  // needed.
  void save(const std::string& fname) {
    torch::serialize::OutputArchive archive;
    archive.write("step_count", torch::IValue((int64_t)step_count_));
    archive.write("lrs", torch::IValue(get_current_lrs()));
    archive.save_to(fname);
  }
  void load(const std::string& fname, torch::optim::Optimizer& optimizer) {
    torch::serialize::InputArchive archive;
    archive.load_from(fname);

    torch::IValue ivalue;
    if (!archive.try_read("step_count", ivalue)) {
      THROW_INVALID_USAGE(fname + " is missing required data.");
    }
    int64_t step_count = ivalue.to<int64_t>();
    step_count_ = step_count;

    if (!archive.try_read("lrs", ivalue)) {
      THROW_INVALID_USAGE(fname + " is missing required data.");
    }
    auto lrs = ivalue.to<std::vector<double>>();
    // Can't use this method to set the LRs due to it being private in the base LR class.
    // set_optimizer_lrs(lrs);
    for (const auto i : c10::irange(optimizer.param_groups().size())) {
      optimizer.param_groups()[i].options().set_lr(lrs[i]);
    }
  }
};

} // namespace torchfort
