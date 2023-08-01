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

#include <algorithm>
#include <vector>

#include "internal/base_lr_scheduler.h"
#include "internal/lr_schedulers.h"

namespace torchfort {

MultiStepLR::MultiStepLR(torch::optim::Optimizer& optimizer, const std::vector<int>& milestones, const double gamma)
    : BaseLRScheduler(optimizer), milestones_(milestones), gamma_(gamma) {}

std::vector<double> MultiStepLR::get_lrs() {
  std::vector<double> lrs = get_current_lrs();
  if (step_count_ == 0 || milestones_.size() == 0)
    return lrs;
  else {
    auto lower_old = std::lower_bound(milestones_.begin(), milestones_.end(), step_count_ - 1,
                                      [](const int& ms, int value) { return ms <= value; });
    auto lower = std::lower_bound(milestones_.begin(), milestones_.end(), step_count_,
                                  [](const int& ms, int value) { return ms <= value; });

    if (lower_old != lower) {
      // in this case we need to decay the LR:
      std::transform(lrs.begin(), lrs.end(), lrs.begin(), [this](const double& lr) { return this->gamma_ * lr; });
    }

    return lrs;
  }
}

} // namespace torchfort
