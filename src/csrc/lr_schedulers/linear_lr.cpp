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

LinearLR::LinearLR(torch::optim::Optimizer& optimizer, const unsigned total_iters, const double start_factor,
                   const double end_factor)
    : BaseLRScheduler(optimizer), total_iters_(total_iters), start_factor_(start_factor), end_factor_(end_factor) {}

std::vector<double> LinearLR::get_lrs() {

  double factor;
  if (step_count_ == 0) {
    factor = start_factor_;
  } else if (step_count_ > total_iters_) {
    factor = 1.;
  } else {
    factor = (1. + (end_factor_ - start_factor_) /
                       double(total_iters_ * start_factor_ + (step_count_ - 1) * (end_factor_ - start_factor_)));
  }

  // get current lrs and modify
  std::vector<double> lrs = get_current_lrs();
  std::transform(lrs.begin(), lrs.end(), lrs.begin(), [factor](const double& v) { return factor * v; });

  return lrs;
}

} // namespace torchfort
