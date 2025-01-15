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

#include "internal/rl/replay_buffer.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace torchfort;
using namespace torch::indexing;

// helper functions
std::shared_ptr<rl::UniformReplayBuffer> getTestReplayBuffer(int buffer_size, int n_envs = 1, float gamma = 0.95, int nstep = 1) {

  torch::NoGradGuard no_grad;

  auto rbuff = std::make_shared<rl::UniformReplayBuffer>(buffer_size, buffer_size, n_envs, gamma, nstep,
                                                         rl::RewardReductionMode::Sum, -1);

  // initialize rng
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1, 5);

  // fill the buffer
  torch::Tensor state = torch::zeros({n_envs,1}, torch::kFloat32), state_p, action, rtens, dtens;
  for (unsigned int i = 0; i < buffer_size; ++i) {
    action = torch::ones({n_envs, 1}, torch::kFloat32) * static_cast<float>(dist(rng));
    state_p = state + action;
    rtens = action.index({"...", 0}).clone();
    // done = false: -> d=0;
    dtens = torch::zeros({n_envs}, torch::kFloat32);
    rbuff->update(state, action, state_p, rtens, dtens);
    state.copy_(state_p);
  }

  return rbuff;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
extract_entries(std::shared_ptr<rl::UniformReplayBuffer> buffp) {
  std::vector<torch::Tensor> slist, alist, splist, rlist, dlist;
  for (unsigned int i = 0; i < buffp->getSize(); ++i) {
    torch::Tensor s, a, sp, r, d;
    std::tie(s, a, sp, r, d) = buffp->get(i);
    slist.push_back(s);
    alist.push_back(a);
    splist.push_back(sp);
    rlist.push_back(r);
    dlist.push_back(d);
  }

  torch::Tensor stens = torch::stack(slist, 0).clone();
  torch::Tensor atens = torch::stack(alist, 0).clone();
  torch::Tensor sptens = torch::stack(splist, 0).clone();
  torch::Tensor rtens = torch::stack(rlist, 0).clone();
  torch::Tensor dtens = torch::stack(dlist, 0).clone();

  return std::make_tuple(stens, atens, sptens, rtens, dtens);
}

void print_buffer(std::shared_ptr<rl::UniformReplayBuffer> buffp) {
  torch::Tensor stens, atens, sptens, reward, d;
  for (int i = 0; i < buffp->getSize(); ++i) {
    std::tie(stens, atens, sptens, reward, d) = buffp->get(i);
    for (int e = 0; e < buffp->nEnvs(); ++e) {
      std::cout << "entry (index, env): (" << i << ", " << e << "): s = " << stens.index({e, "..."}).item<float>() << " a = " << atens.index({e, "..."}).item<float>()
		<< " s' = " << sptens.index({e, "..."}).item<float>() << " r = " << reward.index({e}).item<float>() << " d = " << d.index({e}).item<float>() << std::endl;
    }
  }
}

// check if entries are consistent
TEST(ReplayBuffer, EntryConsistency) {

  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;
  unsigned int n_envs = 1;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, n_envs, 0.95, 1);

  // sample
  torch::Tensor stens, atens, sptens, rtens, dtens;
  float state_diff = 0;
  float reward_diff = 0.;
  for (unsigned int i = 0; i < 4; ++i) {
    std::tie(stens, atens, sptens, rtens, dtens) = rbuff->sample(batch_size);

    // compute differences:
    state_diff += torch::sum(torch::abs(stens + atens - sptens)).item<float>();
    reward_diff += torch::sum(torch::abs(atens.index({"...", 0}) - rtens)).item<float>();
  }

  // success condition
  EXPECT_FLOAT_EQ(state_diff, 0.);
  EXPECT_FLOAT_EQ(reward_diff, 0.);
}

// check if ordering between entries are consistent
TEST(ReplayBuffer, TrajectoryConsistency) {

  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;
  unsigned int n_envs = 1;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, n_envs, 0.95, 1);

  // get a few items and their successors:
  torch::Tensor stens, atens, sptens, sptens_tmp, rtens, dtens;
  // get item at index
  std::tie(stens, atens, sptens, rtens, dtens) = rbuff->get(0);
  // get next item
  float state_diff = 0.;
  for (unsigned int i = 1; i < buffer_size; ++i) {
    std::tie(stens, atens, sptens_tmp, rtens, dtens) = rbuff->get(i);
    state_diff += torch::sum(torch::abs(stens - sptens)).item<float>();
    sptens.copy_(sptens_tmp);
  }

  // success condition
  EXPECT_FLOAT_EQ(state_diff, 0.);
}

// check if nstep reward calculation is correct
TEST(ReplayBuffer, NStepConsistency) {

  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 32;
  unsigned int buffer_size = 8 * batch_size;
  unsigned int n_envs = 1;
  unsigned int nstep = 4;
  float gamma = 0.95;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, n_envs, gamma, nstep);

  // sample a batch
  torch::Tensor stens, atens, sptens, rtens, dtens;
  float state_diff = 0;
  float reward_diff = 0.;
  std::tie(stens, atens, sptens, rtens, dtens) = rbuff->sample(batch_size);
  
  // iterate over samples in batch
  torch::Tensor stemp, atemp, sptemp, sstens, rtemp, dtemp;
  float reward, gamma_eff, rdiff, sdiff, sstens_val;

  // init differences:
  rdiff = 0.;
  sdiff = 0.;
  for (int64_t s = 0; s < batch_size; ++s) {
    sstens = stens.index({s, "..."});
    sstens_val = sstens.item<float>();

    // find the corresponding state
    for (unsigned int i = 0; i < buffer_size; ++i) {
      std::tie(stemp, atemp, sptemp, rtemp, dtemp) = rbuff->get(i);
      if (std::abs(stemp.index({0, "..."}).item<float>() - sstens_val) < 1e-7) {
        // found the right state
        gamma_eff = 1.;
        reward = rtemp.index({0}).item<float>();
        for (unsigned int k = 1; k < nstep; k++) {
          std::tie(stemp, atemp, sptemp, rtemp, dtemp) = rbuff->get(i + k);
          gamma_eff *= gamma;
          reward += rtemp.index({0}).item<float>() * gamma_eff;
        }
        break;
      }
    }
    rdiff += std::abs(reward - rtens.index({s}).item<float>());
    sdiff += torch::sum(torch::abs(sptemp.index({0, "..."}) - sptens.index({s, "..."}))).item<float>();
  }

  EXPECT_FLOAT_EQ(sdiff, 0.);
  EXPECT_FLOAT_EQ(rdiff, 0.);
}

TEST(ReplayBuffer, SaveRestore) {

  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  unsigned int n_envs = 1;
  float gamma = 0.95;
  int nstep = 1;

  // get rollout buffer
  std::shared_ptr<rl::UniformReplayBuffer> rbuff = getTestReplayBuffer(buffer_size, n_envs, gamma, nstep);

  // extract entries before storing
  torch::Tensor stens_b, atens_b, sptens_b, rtens_b, dtens_b;
  std::tie(stens_b, atens_b, sptens_b, rtens_b, dtens_b) = extract_entries(rbuff);

  // store the buffer
  rbuff->save("/tmp/replay_buffer.pt");

  // reset the buffer
  rbuff->reset();

  // reload the buffer
  rbuff->load("/tmp/replay_buffer.pt");

  // extract contents:
  torch::Tensor stens_a, atens_a, sptens_a, rtens_a, dtens_a;
  std::tie(stens_a, atens_a, sptens_a, rtens_a, dtens_a) = extract_entries(rbuff);

  // compute differences:
  float stens_diff = torch::sum(torch::abs(stens_b - stens_a)).item<float>();
  float atens_diff = torch::sum(torch::abs(atens_b - atens_a)).item<float>();
  float sptens_diff = torch::sum(torch::abs(sptens_b - sptens_a)).item<float>();
  float rtens_diff = torch::sum(torch::abs(rtens_b - rtens_a)).item<float>();
  float dtens_diff = torch::sum(torch::abs(dtens_b - dtens_a)).item<float>();

  // success criteria
  EXPECT_FLOAT_EQ(stens_diff, 0.);
  EXPECT_FLOAT_EQ(atens_diff, 0.);
  EXPECT_FLOAT_EQ(sptens_diff, 0.);
  EXPECT_FLOAT_EQ(rtens_diff, 0.);
  EXPECT_FLOAT_EQ(dtens_diff, 0.);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
