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
