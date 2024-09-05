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
#include <vector>

#include <torch/torch.h>

#include "internal/base_lr_scheduler.h"

namespace torchfort {

class CosineAnnealingLR : public BaseLRScheduler {
public:
  CosineAnnealingLR(torch::optim::Optimizer& optimizer, const unsigned T_max, const double eta_min = 0.0);

private:
  std::vector<double> get_lrs() override;
  double update_lr(const double& last_lr, const double& base_lr);

  const unsigned T_max_;
  const double eta_min_;
  std::vector<double> base_lrs_;
};

class MultiStepLR : public BaseLRScheduler {
public:
  MultiStepLR(torch::optim::Optimizer& optimizer, const std::vector<int>& milestones, const double gamma = 0.1);

private:
  std::vector<double> get_lrs() override;

  const std::vector<int> milestones_;
  const double gamma_;
};

class PolynomialLR : public BaseLRScheduler {
public:
  PolynomialLR(torch::optim::Optimizer& optimizer, const unsigned total_iters, const double power = 1.0);

private:
  std::vector<double> get_lrs() override;

  const unsigned total_iters_;
  const double power_;
};

class StepLR : public BaseLRScheduler {
public:
  StepLR(torch::optim::Optimizer& optimizer, const unsigned step_size, const double gamma = 0.1);

private:
  std::vector<double> get_lrs() override;

  const unsigned step_size_;
  const double gamma_;
};

class LinearLR : public BaseLRScheduler {
public:
  LinearLR(torch::optim::Optimizer& optimizer, const unsigned total_iters, const double start_factor = 0.333,
           const double end_factor = 1.0);

private:
  std::vector<double> get_lrs() override;

  const unsigned total_iters_;
  const double start_factor_, end_factor_;
};

} // namespace torchfort
