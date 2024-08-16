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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "internal/rl/rollout_buffer.h"

using namespace torchfort;
using namespace torch::indexing;

// helper functions
std::tuple<std::shared_ptr<rl::GAELambdaRolloutBuffer>, float, bool>
getTestRolloutBuffer(int buffer_size, float gamma=0.95, float lambda=0.99) {

  torch::NoGradGuard no_grad;

  auto rbuff = std::make_shared<rl::GAELambdaRolloutBuffer>(buffer_size, gamma, lambda, -1);

  // initialize rng
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1,5);
  std::normal_distribution<float> normal(1.0, 1.0);

  // fill the buffer
  float	reward, log_p, q;
  bool done;
  torch::Tensor state = torch::zeros({1}, torch::kFloat32), action;
  for (unsigned int i=0; i<buffer_size+1; ++i) {
    action = torch::ones({1}, torch::kFloat32) * static_cast<float>(dist(rng));
    reward = action.item<float>();
    q = reward;
    log_p = normal(rng);
    // add one episode break in the middle. Note that this means that
    // at this index, the state will be the last one in the episode
    // internally this will be converted such that the next state will be the
    // first state in a new episode
    done = (i == (buffer_size / 2) ? true : false);
    rbuff->update(state, action, reward, q, log_p, done);
    state = state + action;
  }

  return std::make_tuple(rbuff, q, done);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
extract_entries(std::shared_ptr<rl::GAELambdaRolloutBuffer> buffp) {
  std::vector<float> svec, avec, rvec, qvec, log_p_vec, advvec, retvec;
  std::vector<float> dvec;
  for (unsigned int i=0; i<buffp->getSize(); ++i) {
    torch::Tensor s, a;
    float r, q, log_p, adv, ret;
    bool d;
    std::tie(s, a, r, q, log_p, adv, ret, d) = buffp->getFull(i);
    svec.push_back(s.item<float>());
    avec.push_back(a.item<float>());
    rvec.push_back(r);
    qvec.push_back(q);
    log_p_vec.push_back(log_p);
    advvec.push_back(adv);
    retvec.push_back(ret);
    dvec.push_back((d ? 1. : 0.));
  }
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor stens = torch::from_blob(svec.data(), {1}, options).clone();
  torch::Tensor atens = torch::from_blob(avec.data(), {1}, options).clone();
  torch::Tensor rtens = torch::from_blob(rvec.data(), {1}, options).clone();
  torch::Tensor qtens = torch::from_blob(qvec.data(), {1}, options).clone();
  torch::Tensor log_p_tens = torch::from_blob(log_p_vec.data(), {1}, options).clone();
  torch::Tensor advtens = torch::from_blob(advvec.data(), {1}, options).clone();
  torch::Tensor rettens = torch::from_blob(retvec.data(), {1}, options).clone();
  torch::Tensor dtens = torch::from_blob(dvec.data(), {1}, options).clone();

  return std::make_tuple(stens, atens, rtens, qtens, log_p_tens, advtens, rettens, dtens);
}

void print_buffer(std::shared_ptr<rl::GAELambdaRolloutBuffer> buffp) {
  torch::Tensor stens, atens;
  float reward, q, log_p;
  bool done;
  for(unsigned int i=0; i<buffp->getSize(); ++i) {
    std::tie(stens, atens, reward, q, log_p, done) = buffp->get(i);
    std::cout << "entry " << i << ": s = " << stens.item<float>() << " a = " << atens.item<float>()
	      << " r = " << reward << " q = " << q << " log_p = " << log_p << " d = " << done << std::endl;
  }
}

// check if entries are consistent
TEST(RolloutBuffer, EntryConsistency) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 2;
  unsigned int buffer_size = 4 * batch_size;
  unsigned int n_iters = 4;
  float gamma = 0.95;
  float lambda = 0.99;

  // get replay buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  float last_val;
  bool last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, gamma, lambda);

  // sample
  torch::Tensor stens, atens, qtens, log_p_tens, advtens, rettens;
  float q_diff = 0.;
  for (unsigned int i=0; i<n_iters; ++i) {
    std::tie(stens, atens, qtens, log_p_tens, advtens, rettens) = rbuff->sample(batch_size);

    // compute differences:
    q_diff += torch::sum(torch::abs(qtens - (rettens - advtens))).item<float>() / static_cast<float>(n_iters);
  }

  // success condition
  EXPECT_NEAR(q_diff, 0., 1e-5);
}

// check if ordering between entries are consistent
TEST(RolloutBuffer, AdvantageComputation) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  float gamma = 0.95;
  float lambda = 0.99;

  // get replay buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  float last_val;
  bool last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, gamma, lambda);

  // get a few items and their successors:
  torch::Tensor stens, atens;
  float r, q, log_p, adv, ret, df;
  bool d;
  torch::Tensor rtens = torch::empty({buffer_size}, torch::kFloat32);
  torch::Tensor advtens = torch::empty({buffer_size}, torch::kFloat32);
  torch::Tensor advtens_compare = torch::empty({buffer_size}, torch::kFloat32);
  // this tensor needs to be a bit bigger because it needs to hold the final q
  torch::Tensor qtens =	torch::empty({buffer_size+1}, torch::kFloat32);
  torch::Tensor dftens = torch::empty({buffer_size+1}, torch::kFloat32);
  // first, extract all V and r elements of the tensor and move them into a big tensor:
  for (int i=0; i<buffer_size; ++i) {
    std::tie(stens, atens, r, q, log_p, adv, ret, d) = rbuff->getFull(i);
    rtens.index_put_({i}, r);
    qtens.index_put_({i}, q);
    df = (d ? 0. : 1.);
    dftens.index_put_({i}, df);
    advtens_compare.index_put_({i}, adv);
  }
  qtens.index_put_({static_cast<int>(buffer_size)}, last_val);
  dftens.index_put_({static_cast<int>(buffer_size)}, (last_done ? 0. : 1.));

  // compute delta
  torch::Tensor deltatens = rtens + dftens.index({Slice(1, buffer_size+1, 1)}) * gamma * qtens.index({Slice(1, buffer_size+1, 1)}) - qtens.index({Slice(0, buffer_size, 1)});

  // compute discounted cumulative sum:
  float delta = deltatens.index({static_cast<int>(buffer_size)-1}).item<float>();
  advtens.index_put_({static_cast<int>(buffer_size)-1}, delta);
  for (int i=(buffer_size-2); i>=0; --i) {
    delta = deltatens.index({i}).item<float>();
    // do not incorporate next entry if new episode starts
    advtens.index_put_({i}, delta +  gamma * lambda * dftens.index({i+1}) * advtens.index({i+1}));
  }

  float adv_diff = torch::sum(advtens_compare - advtens).item<float>();
  EXPECT_FLOAT_EQ(adv_diff, 0.);
}

TEST(RolloutBuffer, SaveRestore) {
  // rng
  torch::manual_seed(666);

  // some parameters
  unsigned int batch_size = 1;
  unsigned int buffer_size = 8 * batch_size;
  float gamma = 0.95;
  float lambda = 0.99;

  // get rollout buffer
  std::shared_ptr<rl::GAELambdaRolloutBuffer> rbuff;
  float last_val;
  bool last_done;
  std::tie(rbuff, last_val, last_done) = getTestRolloutBuffer(buffer_size, gamma, lambda);

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
  float stens_diff = torch::sum(torch::abs(stens_b-stens_a)).item<float>();
  float atens_diff = torch::sum(torch::abs(atens_b-atens_a)).item<float>();
  float rtens_diff = torch::sum(torch::abs(rtens_b-rtens_a)).item<float>();
  float qtens_diff = torch::sum(torch::abs(qtens_b-qtens_a)).item<float>();
  float log_p_tens_diff = torch::sum(torch::abs(log_p_tens_b-log_p_tens_a)).item<float>();
  float advtens_diff = torch::sum(torch::abs(advtens_b-advtens_a)).item<float>();
  float rettens_diff = torch::sum(torch::abs(rettens_b-rettens_a)).item<float>();
  float dtens_diff = torch::sum(torch::abs(dtens_b-dtens_a)).item<float>();

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

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
