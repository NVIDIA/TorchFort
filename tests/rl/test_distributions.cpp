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
