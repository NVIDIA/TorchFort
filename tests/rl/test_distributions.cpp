/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/rl/distributions.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace torchfort;
using namespace torch::indexing;

TEST(NormalDistribution, RandomSampling) {
  // rng
  torch::manual_seed(666);

  // no grad guard
  torch::NoGradGuard no_grad;

  // create normal distribution with given shape
  torch::Tensor mutens = torch::empty({4, 8}, torch::kFloat32);
  torch::Tensor log_sigmatens = torch::empty({4, 8}, torch::kFloat32);

  // fill with random elements
  mutens.normal_();
  log_sigmatens.normal_();
  torch::Tensor sigmatens = torch::exp(log_sigmatens);

  auto ndist = rl::NormalDistribution(mutens, sigmatens);
  torch::Tensor sample = ndist.rsample();

  // do direct sampling without reparametrization trick
  torch::Tensor sample_compare = at::normal(mutens, sigmatens);

  // expect that shapes match: I am not sure how to compare the values as well
  EXPECT_NO_THROW(torch::sum(sample - sample_compare).item<float>());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
