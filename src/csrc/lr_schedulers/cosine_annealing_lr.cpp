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

#include <cmath>

#include "internal/base_lr_scheduler.h"
#include "internal/lr_schedulers.h"

namespace torchfort {

CosineAnnealingLR::CosineAnnealingLR(torch::optim::Optimizer& optimizer, const unsigned T_max, const double eta_min)
    : BaseLRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min) {
  base_lrs_ = get_current_lrs();
}

double CosineAnnealingLR::update_lr(const double& last_lr, const double& base_lr) {
  double lr;
  if ((step_count_ - 1 - T_max_) % (2 * T_max_) == 0) {
    lr = eta_min_ + 0.5 * (base_lr - eta_min_) * (1. + cos(double(step_count_) * M_PI / double(T_max_)));
  } else {
    lr = (1. + cos(M_PI * double(step_count_) / double(T_max_))) /
             (1. + cos(M_PI * double(step_count_ - 1) / double(T_max_))) * (last_lr - eta_min_) +
         eta_min_;
  }

  return lr;
}

std::vector<double> CosineAnnealingLR::get_lrs() {
  std::vector<double> lrs = get_current_lrs();
  if (step_count_ == 0 || T_max_ == 0)
    return lrs;
  else {
    std::vector<double> lrs_new;
    std::transform(lrs.begin(), lrs.end(), base_lrs_.begin(), std::back_inserter(lrs_new),
                   [this](const auto& current, const auto& base) { return update_lr(current, base); });
    return lrs_new;
  }
}

} // namespace torchfort
