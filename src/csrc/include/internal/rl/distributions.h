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
