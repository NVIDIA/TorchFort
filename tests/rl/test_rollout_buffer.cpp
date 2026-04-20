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

#include "internal/rl/rollout_buffer.h"
#include "internal/rl/running_normalizer.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace torchfort;
using namespace torch::indexing;

class RolloutBuffer : public testing::TestWithParam<int> {};

// helper functions
std::tuple<std::shared_ptr<rl::GAELambdaRolloutBuffer>, torch::Tensor, torch::Tensor>
getTestRolloutBuffer(int buffer_size, int n_env, float gamma = 0.95, float lambda = 0.99) {

  torch::NoGradGuard no_grad;

  auto rbuff = std::make_shared<rl::GAELambdaRolloutBuffer>(buffer_size, n_env, gamma, lambda, -1);

  // initialize rng
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1, 5);
  std::normal_distribution<float> normal(1.0, 1.0);

  // fill the buffer
  torch::Tensor state = torch::zeros({n_env, 1}, torch::kFloat32);
  torch::Tensor action, reward, log_p, q, done;
  for (int i = 0; i < (buffer_size / n_env) + 1; ++i) {
    action = torch::ones({n_env, 1}, torch::kFloat32);
    for (int e = 0; e < n_env; ++e) {
      action.index_put_({e, 0}, static_cast<float>(dist(rng)));
    }
    reward = torch::squeeze(action, 1).clone();
    q = reward.clone();
    log_p = torch::ones({n_env}, torch::kFloat32) * normal(rng);
    // add one episode break in the middle. Note that this means that
    // at this index, the state will be the last one in the episode
    // internally this will be converted such that the next state will be the
    // first state in a new episode
    done = torch::ones({n_env}, torch::kFloat32) * (i == (buffer_size / 2) ? 1. : 0.);
    rbuff->update(state, action, reward, q, log_p, done);
    state = state + action;
  }

  return std::make_tuple(rbuff, q, done);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
extract_entries(std::shared_ptr<rl::GAELambdaRolloutBuffer> buffp) {
  std::vector<torch::Tensor> svec, avec, rvec, qvec, log_p_vec, advvec, retvec, dvec;
  unsigned int max_iter = (buffp->getSize()) / (buffp->nEnvs());
  for (unsigned int i = 0; i < max_iter; ++i) {
    torch::Tensor s, a, r, q, log_p, adv, ret, d;
    std::tie(s, a, r, q, log_p, adv, ret, d) = buffp->getFull(i);
    svec.push_back(s);
    avec.push_back(a);
    rvec.push_back(r);
    qvec.push_back(q);
    log_p_vec.push_back(log_p);
    advvec.push_back(adv);
    retvec.push_back(ret);
    dvec.push_back(d);
  }
  torch::Tensor stens = torch::cat(svec, 0);
  torch::Tensor atens = torch::cat(avec, 0);
  torch::Tensor rtens = torch::cat(rvec, 0);
  torch::Tensor qtens = torch::cat(qvec, 0);
  torch::Tensor log_p_tens = torch::cat(log_p_vec, 0);
  torch::Tensor advtens = torch::cat(advvec, 0);
  torch::Tensor rettens = torch::cat(retvec, 0);
  torch::Tensor dtens = torch::cat(dvec, 0);

  return std::make_tuple(stens, atens, rtens, qtens, log_p_tens, advtens, rettens, dtens);
}

void print_buffer(std::shared_ptr<rl::GAELambdaRolloutBuffer> buffp) {
  torch::Tensor stens, atens, reward, q, log_p, done;
  for (unsigned int i = 0; i < buffp->getSize(); ++i) {
    std::tie(stens, atens, reward, q, log_p, done) = buffp->get(i);
    std::cout << "entry " << i << ": s = " << stens.index({0, 0}).item<float>()
              << " a = " << atens.index({0, 0}).item<float>() << " r = " << reward.item<float>()
              << " q = " << q.item<float>() << " log_p = " << log_p.item<float>()
              << " d = " << (done.item<float>() > 0.5 ? true : false) << std::endl;
  }
}

// check if shapes match expected shapes
TEST_P(RolloutBuffer, ShapeConsistency) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_env = GetParam();
  unsigned int batch_size = 2;
  unsigned int buffer_size = 4 * batch_size;
  unsigned int n_iters = 4;
  float gamma = 0.95;
  float lambda = 0.99;

  // get replay buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env, gamma, lambda);

  // sample
  torch::Tensor stens, atens, qtens, log_p_tens, advtens, rettens;

  std::tie(stens, atens, qtens, log_p_tens, advtens, rettens) = rbuff->sample(batch_size);

  // check shapes
  EXPECT_EQ(stens.dim(), 2);
  EXPECT_EQ(atens.dim(), 2);
  EXPECT_EQ(qtens.dim(), 1);
  EXPECT_EQ(log_p_tens.dim(), 1);
  EXPECT_EQ(advtens.dim(), 1);
  EXPECT_EQ(rettens.dim(), 1);

  EXPECT_EQ(stens.size(0), batch_size);
  EXPECT_EQ(atens.size(0), batch_size);
  EXPECT_EQ(qtens.size(0), batch_size);
  EXPECT_EQ(log_p_tens.size(0), batch_size);
  EXPECT_EQ(advtens.size(0), batch_size);
  EXPECT_EQ(rettens.size(0), batch_size);

  EXPECT_EQ(stens.size(1), 1);
  EXPECT_EQ(atens.size(1), 1);
}

// check if entries are consistent
TEST_P(RolloutBuffer, EntryConsistency) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_env = GetParam();
  unsigned int batch_size = 2;
  unsigned int buffer_size = 4 * batch_size;
  unsigned int n_iters = 4;
  float gamma = 0.95;
  float lambda = 0.99;

  // get replay buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env, gamma, lambda);

  // sample
  torch::Tensor stens, atens, qtens, log_p_tens, advtens, rettens;
  float q_diff = 0.;
  for (unsigned int i = 0; i < n_iters; ++i) {
    std::tie(stens, atens, qtens, log_p_tens, advtens, rettens) = rbuff->sample(batch_size);

    // compute differences:
    q_diff += torch::sum(torch::abs(qtens - (rettens - advtens))).item<float>() / static_cast<float>(n_iters);
  }

  // success condition
  EXPECT_NEAR(q_diff, 0., 1e-5);
}

// check if ordering between entries are consistent
TEST_P(RolloutBuffer, AdvantageComputation) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_env = GetParam();
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  unsigned int eff_buffer_size = (buffer_size / n_env);
  float gamma = 0.95;
  float lambda = 0.99;

  // get replay buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env, gamma, lambda);

  // get a few items and their successors:
  torch::Tensor stens, atens, r, q, log_p, adv, ret, d;
  std::vector<torch::Tensor> rvec, qvec, dfvec, advvec;
  // first, extract all V and r elements of the tensor and move them into a big tensor:
  for (int i = 0; i < eff_buffer_size; ++i) {
    std::tie(stens, atens, r, q, log_p, adv, ret, d) = rbuff->getFull(i);
    rvec.push_back(r);
    qvec.push_back(q);
    dfvec.push_back(1. - d);
    advvec.push_back(adv);
  }
  qvec.push_back(last_val);
  dfvec.push_back(1. - last_done);

  torch::Tensor rtens = torch::stack(rvec, 0);
  torch::Tensor qtens = torch::stack(qvec, 0);
  torch::Tensor dftens = torch::stack(dfvec, 0);
  torch::Tensor advtens_compare = torch::stack(advvec, 0);
  torch::Tensor advtens = torch::zeros_like(advtens_compare);

  // compute delta
  torch::Tensor deltatens = rtens +
                            dftens.index({Slice(1, eff_buffer_size + 1, 1), "..."}) * gamma *
                                qtens.index({Slice(1, eff_buffer_size + 1, 1), "..."}) -
                            qtens.index({Slice(0, eff_buffer_size, 1), "..."});

  // compute discounted cumulative sum:
  torch::Tensor delta = deltatens.index({static_cast<int>(eff_buffer_size) - 1, "..."});
  advtens.index({static_cast<int>(eff_buffer_size) - 1, "..."}) = delta.index({"..."});
  for (int i = (eff_buffer_size - 2); i >= 0; --i) {
    delta = deltatens.index({i, "..."});
    // do not incorporate next entry if new episode starts
    advtens.index({i, "..."}) = delta + gamma * lambda * dftens.index({i + 1, "..."}) * advtens.index({i + 1, "..."});
  }

  float adv_diff = torch::sum(advtens_compare - advtens).item<float>();
  EXPECT_FLOAT_EQ(adv_diff, 0.);
}

TEST_P(RolloutBuffer, SaveRestore) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int n_env = GetParam();
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  float gamma = 0.95;
  float lambda = 0.99;

  // get rollout buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env, gamma, lambda);

  // extract entries before storing
  torch::Tensor stens_b, atens_b, rtens_b, qtens_b, log_p_tens_b, advtens_b, rettens_b, dtens_b;
  std::tie(stens_b, atens_b, rtens_b, qtens_b, log_p_tens_b, advtens_b, rettens_b, dtens_b) = extract_entries(rbuff);

  // store the buffer
  rbuff->save("/tmp/rollout_buffer.pt");

  // reset the buffer
  rbuff->reset();

  // reload the buffer
  rbuff->load("/tmp/rollout_buffer.pt");

  // extract contents:
  torch::Tensor stens_a, atens_a, rtens_a, qtens_a, log_p_tens_a, advtens_a, rettens_a, dtens_a;
  std::tie(stens_a, atens_a, rtens_a, qtens_a, log_p_tens_a, advtens_a, rettens_a, dtens_a) = extract_entries(rbuff);

  // compute differences:
  float stens_diff = torch::sum(torch::abs(stens_b - stens_a)).item<float>();
  float atens_diff = torch::sum(torch::abs(atens_b - atens_a)).item<float>();
  float rtens_diff = torch::sum(torch::abs(rtens_b - rtens_a)).item<float>();
  float qtens_diff = torch::sum(torch::abs(qtens_b - qtens_a)).item<float>();
  float log_p_tens_diff = torch::sum(torch::abs(log_p_tens_b - log_p_tens_a)).item<float>();
  float advtens_diff = torch::sum(torch::abs(advtens_b - advtens_a)).item<float>();
  float rettens_diff = torch::sum(torch::abs(rettens_b - rettens_a)).item<float>();
  float dtens_diff = torch::sum(torch::abs(dtens_b - dtens_a)).item<float>();

  // success criteria
  EXPECT_FLOAT_EQ(stens_diff, 0.);
  EXPECT_FLOAT_EQ(atens_diff, 0.);
  EXPECT_FLOAT_EQ(rtens_diff, 0.);
  EXPECT_FLOAT_EQ(qtens_diff, 0.);
  EXPECT_FLOAT_EQ(log_p_tens_diff, 0.);
  EXPECT_FLOAT_EQ(advtens_diff, 0.);
  EXPECT_FLOAT_EQ(rettens_diff, 0.);
  EXPECT_FLOAT_EQ(dtens_diff, 0.);
}

INSTANTIATE_TEST_SUITE_P(MultiEnv, RolloutBuffer, testing::Range(1, 3), testing::PrintToStringParamName());

// =========================================================================
// normalizeReturns tests
// These tests use n_env=1 for simplicity; the multi-env path is covered by
// the parameterized suite above for the base buffer operations.
// =========================================================================

// ---- NormalizeReturns: A = R - V relationship is preserved ---------------
// normalizeReturns scales both returns and advantages by the same factor, so
// the relationship A = R - V must still hold exactly after normalization.
TEST(NormalizeReturns, MaintainsAdvantageReturnRelationship) {
  torch::manual_seed(42);
  torch::NoGradGuard no_grad;

  const int buffer_size = 16;
  const int n_env = 1;

  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env);

  // apply return normalization
  rl::RunningNormalizer normalizer(1e-8f, /* scale_only = */ true);
  rbuff->normalizeReturns(nullptr, normalizer);

  // verify A = R - V still holds for every entry
  float max_violation = 0.f;
  int n_steps = buffer_size / n_env;
  for (int i = 0; i < n_steps; ++i) {
    torch::Tensor s, a, r, q, log_p, adv, ret, d;
    std::tie(s, a, r, q, log_p, adv, ret, d) = rbuff->getFull(i);
    // ret = adv + q  =>  adv - (ret - q) should be ~0
    float violation = torch::sum(torch::abs(adv - (ret - q))).item<float>();
    max_violation = std::max(max_violation, violation);
  }

  EXPECT_NEAR(max_violation, 0.f, 1e-5f)
      << "A = R - V must hold after normalizeReturns (both scaled by same factor)";
}

// ---- NormalizeReturns: unit std, nonzero mean ----------------------------
// After normalization the collection of all returns should have std ~1 but
// mean should NOT be zero (scale_only preserves the mean).
TEST(NormalizeReturns, UnitStdPreservedMean) {
  torch::manual_seed(43);
  torch::NoGradGuard no_grad;

  // Use a larger buffer to get a stable std estimate
  const int buffer_size = 128;
  const int n_env = 1;

  // Warm up the normalizer over several rollouts so it has stable stats
  rl::RunningNormalizer normalizer(1e-8f, /* scale_only = */ true);
  for (int rollout = 0; rollout < 20; ++rollout) {
    std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
    torch::Tensor last_val, last_done;
    std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env);
    rbuff->normalizeReturns(nullptr, normalizer);
  }

  // Final rollout: check statistics of normalized returns
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env);
  rbuff->normalizeReturns(nullptr, normalizer);

  // collect all normalized returns
  std::vector<torch::Tensor> ret_vec;
  int n_steps = buffer_size / n_env;
  for (int i = 0; i < n_steps; ++i) {
    torch::Tensor s, a, r, q, log_p, adv, ret, d;
    std::tie(s, a, r, q, log_p, adv, ret, d) = rbuff->getFull(i);
    ret_vec.push_back(ret);
  }
  auto all_ret = torch::cat(ret_vec, 0).flatten().to(torch::kFloat32);

  // std should be ~1 (scale normalization)
  float out_std = all_ret.std().item<float>();
  EXPECT_NEAR(out_std, 1.0f, 0.2f)
      << "Normalized returns should have std ~1";

  // mean should NOT be zero (scale_only: mean is preserved)
  // The test buffer uses positive rewards (dist uniform in [1,5]) so returns > 0
  float out_mean = all_ret.mean().item<float>();
  EXPECT_GT(out_mean, 0.1f)
      << "Normalized returns should have nonzero mean (scale_only preserves mean)";
}

// ---- NormalizeReturns + NormalizeAdvantages: correct combined effect ------
// When both are applied in order (returns first, advantages second), the
// end state should be: returns have unit std + nonzero mean, advantages
// have unit std + zero mean.
TEST(NormalizeReturns, OrderWithAdvantageNormalization) {
  torch::manual_seed(44);
  torch::NoGradGuard no_grad;

  const int buffer_size = 64;
  const int n_env = 1;

  // Warm up the return normalizer
  rl::RunningNormalizer ret_normalizer(1e-8f, /* scale_only = */ true);
  for (int rollout = 0; rollout < 10; ++rollout) {
    std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
    torch::Tensor last_val, last_done;
    std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env);
    rbuff->normalizeReturns(nullptr, ret_normalizer);
  }

  // Final rollout: apply both normalizations in the correct order
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  torch::Tensor last_val, last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, n_env);

  rbuff->normalizeReturns(nullptr, ret_normalizer);   // step 1: scale R and A by return std
  rbuff->normalizeAdvantages(nullptr);                 // step 2: zero-center and unit-std A

  // collect normalized returns and advantages
  std::vector<torch::Tensor> ret_vec, adv_vec;
  int n_steps = buffer_size / n_env;
  for (int i = 0; i < n_steps; ++i) {
    torch::Tensor s, a, r, q, log_p, adv, ret, d;
    std::tie(s, a, r, q, log_p, adv, ret, d) = rbuff->getFull(i);
    ret_vec.push_back(ret);
    adv_vec.push_back(adv);
  }
  auto all_ret = torch::cat(ret_vec, 0).flatten().to(torch::kFloat32);
  auto all_adv = torch::cat(adv_vec, 0).flatten().to(torch::kFloat32);

  // returns: unit std, nonzero mean
  EXPECT_NEAR(all_ret.std().item<float>(), 1.0f, 0.2f)
      << "Returns should have std ~1 after normalizeReturns";
  EXPECT_GT(all_ret.mean().item<float>(), 0.1f)
      << "Returns should have nonzero mean after normalizeReturns (scale_only)";

  // advantages: unit std, zero mean
  EXPECT_NEAR(all_adv.std().item<float>(), 1.0f, 0.1f)
      << "Advantages should have std ~1 after normalizeAdvantages";
  EXPECT_NEAR(all_adv.mean().item<float>(), 0.0f, 0.1f)
      << "Advantages should have zero mean after normalizeAdvantages";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
