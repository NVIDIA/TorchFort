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

#include "internal/rl/replay_buffer.h"
#include "internal/rl/running_normalizer.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace torchfort;
using namespace torch::indexing;

class ReplayBuffer : public testing::TestWithParam<int> {};

// helper functions
std::shared_ptr<rl::UniformReplayBuffer> getTestReplayBuffer(int buffer_size, int n_envs = 1, float gamma = 0.95,
                                                             int nstep = 1) {

  torch::NoGradGuard no_grad;

  auto rbuff = std::make_shared<rl::UniformReplayBuffer>(buffer_size, buffer_size, n_envs, gamma, nstep,
                                                         rl::RewardReductionMode::Sum, -1);

  // initialize rng
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1, 5);

  // fill the buffer
  torch::Tensor state = torch::zeros({n_envs, 1}, torch::kFloat32), state_p, action, rtens, dtens;
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
      std::cout << "entry (index, env): (" << i << ", " << e << "): s = " << stens.index({e, "..."}).item<float>()
                << " a = " << atens.index({e, "..."}).item<float>()
                << " s' = " << sptens.index({e, "..."}).item<float>() << " r = " << reward.index({e}).item<float>()
                << " d = " << d.index({e}).item<float>() << std::endl;
    }
  }
}

// check if shapes match expected shapes
TEST_P(ReplayBuffer, ShapeConsistency) {
  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_envs = GetParam();
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, n_envs, 0.95, 1);

  // sample
  torch::Tensor stens, atens, sptens, rtens, dtens;
  std::tie(stens, atens, sptens, rtens, dtens) = rbuff->sample(batch_size);

  // check shapes
  EXPECT_EQ(stens.dim(), 2);
  EXPECT_EQ(atens.dim(), 2);
  EXPECT_EQ(sptens.dim(), 2);
  EXPECT_EQ(rtens.dim(), 1);
  EXPECT_EQ(dtens.dim(), 1);

  EXPECT_EQ(stens.size(0), batch_size);
  EXPECT_EQ(atens.size(0), batch_size);
  EXPECT_EQ(sptens.size(0), batch_size);
  EXPECT_EQ(rtens.size(0), batch_size);
  EXPECT_EQ(dtens.size(0), batch_size);

  EXPECT_EQ(stens.size(1), 1);
  EXPECT_EQ(atens.size(1), 1);
}

// check if entries are consistent
TEST_P(ReplayBuffer, EntryConsistency) {

  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_envs = GetParam();
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;

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
TEST_P(ReplayBuffer, TrajectoryConsistency) {

  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_envs = GetParam();
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, n_envs, 0.95, 1);

  // get a few items and their successors:
  torch::Tensor stens, atens, sptens, sptens_tmp, rtens, dtens;

  // get item at index
  std::tie(stens, atens, sptens, rtens, dtens) = rbuff->get(0);
  // get next item
  float state_diff = 0.;
  for (unsigned int i = 1; i < rbuff->getSize(); ++i) {
    std::tie(stens, atens, sptens_tmp, rtens, dtens) = rbuff->get(i);
    state_diff += torch::sum(torch::abs(stens - sptens)).item<float>();
    sptens.copy_(sptens_tmp);
  }

  // success condition
  EXPECT_FLOAT_EQ(state_diff, 0.);
}

// check if nstep reward calculation is correct
TEST_P(ReplayBuffer, NStepConsistency) {

  // set rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_envs = GetParam();
  unsigned int batch_size = 32;
  unsigned int buffer_size = 8 * batch_size;
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
  torch::Tensor stemp, atemp, sptemp, sstens, rtemp, dtemp, spfin;
  float reward, gamma_eff, rdiff, sdiff, sstens_val;

  // init differences:
  rdiff = 0.;
  sdiff = 0.;
  for (int64_t s = 0; s < batch_size; ++s) {
    sstens = stens.index({s, "..."});
    sstens_val = sstens.item<float>();

    // find the corresponding state
    bool found = false;
    for (unsigned int i = 0; i < buffer_size; ++i) {
      std::tie(stemp, atemp, sptemp, rtemp, dtemp) = rbuff->get(i);
      for (int64_t e = 0; e < n_envs; ++e) {
        if (std::abs(stemp.index({e, "..."}).item<float>() - sstens_val) < 1e-7) {
          // found the right state
          found = true;
          gamma_eff = 1.;
          reward = rtemp.index({e}).item<float>();
          for (unsigned int k = 1; k < nstep; k++) {
            std::tie(stemp, atemp, sptemp, rtemp, dtemp) = rbuff->get(i + k);
            gamma_eff *= gamma;
            reward += rtemp.index({e}).item<float>() * gamma_eff;
            spfin = sptemp.index({e, "..."});
          }
          break;
        }
      }
      if (found)
        break;
    }
    rdiff += std::abs(reward - rtens.index({s}).item<float>());
    sdiff += torch::sum(torch::abs(spfin - sptens.index({s, "..."}))).item<float>();
  }

  EXPECT_FLOAT_EQ(sdiff, 0.);
  EXPECT_FLOAT_EQ(rdiff, 0.);
}

TEST_P(ReplayBuffer, SaveRestore) {

  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_envs = GetParam();
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
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

INSTANTIATE_TEST_SUITE_P(MultiEnv, ReplayBuffer, testing::Range(1, 3), testing::PrintToStringParamName());

// =========================================================================
// Reward normalization tests
// Simulate the workflow used by DDPG/TD3/SAC: update the reward normalizer
// with each incoming reward batch (as in updateReplayBuffer), then normalize
// rewards sampled from the buffer (as in trainStep).
// =========================================================================

// ---- RewardNormalization: unit std, nonzero mean -------------------------
// After the normalizer has seen enough rewards, normalizing a sampled reward
// batch should yield unit std but preserve the mean (scale_only=true).
TEST(RewardNormalization, UnitStdPreservedMean) {
  torch::manual_seed(300);
  torch::NoGradGuard no_grad;

  // Rewards ~ N(mean=5, std=2): a typical dense-reward task distribution
  const float true_mean = 5.0f;
  const float true_std  = 2.0f;
  const int   n_envs    = 1;
  const int   buffer_size = 512;

  auto rbuff = std::make_shared<rl::UniformReplayBuffer>(
      buffer_size, buffer_size, n_envs, 0.99f, 1, rl::RewardReductionMode::Sum, -1);

  rl::RunningNormalizer reward_normalizer(1e-8f, /* scale_only = */ true);

  // Fill the buffer, updating the normalizer with each reward batch exactly
  // as DDPGSystem::updateReplayBuffer does
  torch::Tensor state = torch::zeros({n_envs, 4}, torch::kFloat32);
  for (int i = 0; i < buffer_size; ++i) {
    auto action = torch::zeros({n_envs, 2}, torch::kFloat32);
    auto next_state = state + 0.01f;
    auto reward = torch::randn({n_envs}, torch::kFloat32) * true_std + true_mean;
    auto done   = torch::zeros({n_envs}, torch::kFloat32);

    // mirror the system's updateReplayBuffer call
    reward_normalizer.update(reward.unsqueeze(1));
    rbuff->update(state, action, next_state, reward, done);
    state = next_state;
  }

  // Sample a batch and normalize rewards as trainStep does
  const int batch_size = 256;
  torch::Tensor s, a, sp, r, d;
  std::tie(s, a, sp, r, d) = rbuff->sample(batch_size);

  // mirror the system's trainStep normalization
  auto r_norm = reward_normalizer.normalize(r.unsqueeze(1)).squeeze(1);

  // std of normalized rewards should be ~1
  EXPECT_NEAR(r_norm.std().item<float>(), 1.0f, 0.15f)
      << "Normalized rewards should have std ~1";

  // mean should be ~true_mean / true_std = 2.5 (not ~0)
  float expected_mean = true_mean / true_std;
  EXPECT_NEAR(r_norm.mean().item<float>(), expected_mean, 0.3f)
      << "Normalized rewards should preserve mean as ~true_mean/true_std";
}

// ---- RewardNormalization: sign preservation ------------------------------
// Rewards from a strictly positive distribution must remain positive after
// normalization. This is the key correctness requirement for scale_only mode.
TEST(RewardNormalization, SignPreservation) {
  torch::manual_seed(301);
  torch::NoGradGuard no_grad;

  // Rewards ~ Uniform(1, 5): always positive
  const int n_envs    = 1;
  const int buffer_size = 256;

  auto rbuff = std::make_shared<rl::UniformReplayBuffer>(
      buffer_size, buffer_size, n_envs, 0.99f, 1, rl::RewardReductionMode::Sum, -1);

  rl::RunningNormalizer reward_normalizer(1e-8f, /* scale_only = */ true);

  torch::Tensor state = torch::zeros({n_envs, 4}, torch::kFloat32);
  for (int i = 0; i < buffer_size; ++i) {
    auto action = torch::zeros({n_envs, 2}, torch::kFloat32);
    auto next_state = state + 0.01f;
    // strictly positive rewards in [1, 5]
    auto reward = torch::rand({n_envs}, torch::kFloat32) * 4.0f + 1.0f;
    auto done   = torch::zeros({n_envs}, torch::kFloat32);

    reward_normalizer.update(reward.unsqueeze(1));
    rbuff->update(state, action, next_state, reward, done);
    state = next_state;
  }

  torch::Tensor s, a, sp, r, d;
  std::tie(s, a, sp, r, d) = rbuff->sample(buffer_size);
  auto r_norm = reward_normalizer.normalize(r.unsqueeze(1)).squeeze(1);

  // all normalized rewards must remain positive
  EXPECT_TRUE((r_norm > 0).all().item<bool>())
      << "All positive rewards must remain positive after scale_only normalization";
}

// ---- RewardNormalization: large-scale rewards normalized to unit range ----
// Rewards with large magnitude (e.g. N(100, 20)) should be brought to unit
// std. Without normalization these would dominate the Bellman target.
TEST(RewardNormalization, LargeScaleNormalizedToUnitStd) {
  torch::manual_seed(302);
  torch::NoGradGuard no_grad;

  const float true_mean = 100.0f;
  const float true_std  = 20.0f;
  const int   n_envs    = 1;
  const int   buffer_size = 512;

  auto rbuff = std::make_shared<rl::UniformReplayBuffer>(
      buffer_size, buffer_size, n_envs, 0.99f, 1, rl::RewardReductionMode::Sum, -1);

  rl::RunningNormalizer reward_normalizer(1e-8f, /* scale_only = */ true);

  torch::Tensor state = torch::zeros({n_envs, 4}, torch::kFloat32);
  for (int i = 0; i < buffer_size; ++i) {
    auto action = torch::zeros({n_envs, 2}, torch::kFloat32);
    auto next_state = state + 0.01f;
    auto reward = torch::randn({n_envs}, torch::kFloat32) * true_std + true_mean;
    auto done   = torch::zeros({n_envs}, torch::kFloat32);

    reward_normalizer.update(reward.unsqueeze(1));
    rbuff->update(state, action, next_state, reward, done);
    state = next_state;
  }

  torch::Tensor s, a, sp, r, d;
  std::tie(s, a, sp, r, d) = rbuff->sample(buffer_size);
  auto r_norm = reward_normalizer.normalize(r.unsqueeze(1)).squeeze(1);

  // std should be close to 1 regardless of the original reward scale
  EXPECT_NEAR(r_norm.std().item<float>(), 1.0f, 0.15f)
      << "Large-scale rewards must be normalized to unit std";

  // mean should be ~true_mean/true_std = 5, not 0 and not 100
  float expected_mean = true_mean / true_std;
  EXPECT_NEAR(r_norm.mean().item<float>(), expected_mean, 0.5f)
      << "Mean should be preserved as ~true_mean/true_std, not removed";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
