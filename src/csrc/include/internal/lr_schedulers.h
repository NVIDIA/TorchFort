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
