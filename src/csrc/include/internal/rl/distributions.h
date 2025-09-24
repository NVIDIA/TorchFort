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
#include <cmath>
#include <torch/torch.h>

#include "internal/rl/rl.h"

namespace torchfort {

namespace rl {

class Distribution {

public:
  Distribution(const Distribution&) = delete;

  // constructor
  Distribution() {}
  virtual torch::Tensor rsample() = 0;
  virtual torch::Tensor log_prob(torch::Tensor value) = 0;
  virtual torch::Tensor entropy() = 0;
};

class NormalDistribution : public Distribution, public std::enable_shared_from_this<Distribution> {
public:
  NormalDistribution(torch::Tensor mu, torch::Tensor sigma) : mu_(mu), sigma_(sigma) {}

  torch::Tensor rsample() {
    auto noise = torch::empty_like(mu_).normal_(0., 1.);
    return torch::Tensor(mu_ + sigma_ * noise).clone();
  }

  torch::Tensor log_prob(torch::Tensor value) {
    auto var = torch::square(sigma_);
    auto log_sigma = sigma_.log();
    auto result = -torch::square(value - mu_) / (2 * var) - log_sigma - std::log(std::sqrt(2. * M_PI));

    return result;
  }

  torch::Tensor entropy() {
    auto log_sigma = sigma_.log();
    auto result = log_sigma + 0.5 * (1. + std::log(2. * M_PI));

    return result;
  }

protected:
  torch::Tensor mu_;
  torch::Tensor sigma_;
};

} // namespace rl

} // namespace torchfort
