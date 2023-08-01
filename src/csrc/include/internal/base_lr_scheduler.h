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
