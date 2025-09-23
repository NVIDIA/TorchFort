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
#include <cmath>
#include <vector>

#include "internal/base_lr_scheduler.h"
#include "internal/lr_schedulers.h"

namespace torchfort {

PolynomialLR::PolynomialLR(torch::optim::Optimizer& optimizer, const unsigned total_iters, const double power)
    : BaseLRScheduler(optimizer), total_iters_(total_iters), power_(power) {}

std::vector<double> PolynomialLR::get_lrs() {
  std::vector<double> lrs = get_current_lrs();
  if (step_count_ == 0 || step_count_ > total_iters_)
    return lrs;
  else {
    double decay_factor =
        (1. - double(step_count_) / double(total_iters_)) / (1. - double(step_count_ - 1) / double(total_iters_));
    decay_factor = std::pow(decay_factor, power_);

    std::transform(lrs.begin(), lrs.end(), lrs.begin(), [decay_factor](const double& v) { return decay_factor * v; });

    return lrs;
  }
}

} // namespace torchfort
