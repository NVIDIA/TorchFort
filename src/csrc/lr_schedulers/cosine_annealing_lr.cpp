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
