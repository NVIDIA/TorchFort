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

#include "internal/rl/rollout_buffer.h"
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
      action.index_put_({e,0}, static_cast<float>(dist(rng)));
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    std::cout << "entry " << i << ": s = " << stens.index({0,0}).item<float>() << " a = " << atens.index({0,0}).item<float>()
	      << " r = " << reward.item<float>() << " q = " << q.item<float>() << " log_p = " << log_p.item<float>()
	      << " d = " << (done.item<float>() > 0.5 ? true : false) << std::endl;
  }
}

// check if entries are consistent
TEST_P(RolloutBuffer, EntryConsistency) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 2;
  unsigned int buffer_size = 4 * batch_size;
  unsigned int n_iters = 4;
  unsigned int n_env = GetParam();
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
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  unsigned int n_env = GetParam();
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
  torch::Tensor	qtens =	torch::stack(qvec, 0);
  torch::Tensor	dftens = torch::stack(dfvec, 0);
  torch::Tensor	advtens_compare = torch::stack(advvec, 0);
  torch::Tensor advtens = torch::zeros_like(advtens_compare);

  // compute delta
  torch::Tensor deltatens =
    rtens + dftens.index({Slice(1, eff_buffer_size + 1, 1), "..."}) * gamma * qtens.index({Slice(1, eff_buffer_size + 1, 1), "..."}) -
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
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  unsigned int n_env = GetParam();
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

INSTANTIATE_TEST_SUITE_P(MultiEnv, RolloutBuffer, testing::Range(1, 3),
                         testing::PrintToStringParamName());

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
