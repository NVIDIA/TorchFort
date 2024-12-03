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

#include "environments.h"
#include "torchfort.h"
#include <gtest/gtest.h>
#include <sstream>

enum EnvMode { Constant, Predictable, Delayed, Action, ActionState };

struct CoutRedirect {
  CoutRedirect(std::streambuf* new_buffer) : old(std::cout.rdbuf(new_buffer)) {}

  ~CoutRedirect() { std::cout.rdbuf(old); }

private:
  std::streambuf* old;
};

std::tuple<float, float, float> TestSystem(const EnvMode mode, const std::string& system,
                                           unsigned int num_explore_iters, unsigned int num_exploit_iters,
                                           unsigned int num_eval_iters = 100, unsigned int num_grad_steps = 1,
                                           bool verbose = false) {

  // capture stdout
  std::stringstream buffer;

  std::shared_ptr<CoutRedirect> red;
  if (!verbose) {
    red = std::make_shared<CoutRedirect>(buffer.rdbuf());
  }

  // set seed
  torch::manual_seed(666);

  // create dummy env:
  std::vector<int64_t> state_shape{1};
  std::vector<int64_t> action_shape{1};
  std::vector<int64_t> state_batch_shape{1, 1};
  std::vector<int64_t> action_batch_shape{1, 1};
  std::vector<int64_t> reward_batch_shape{1};
  float reward, q_estimate, p_loss, q_loss;
  bool done;
  bool is_ready;
  float qdiff = 0.;
  float running_reward = 0.;

  // set up tensors:
  torch::Tensor state = torch::zeros(state_shape, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor state_new = torch::empty_like(state);
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));

  // set up environment
  std::shared_ptr<Environment> env;
  int num_episodes, num_train_iters, episode_length;
  if (mode == Constant) {
    episode_length = 1;
    env = std::make_shared<ConstantRewardEnvironment>(episode_length, state_shape, action_shape, 1.);
  } else if (mode == Predictable) {
    episode_length = 1;
    env = std::make_shared<PredictableRewardEnvironment>(episode_length, state_shape, action_shape);
  } else if (mode == Delayed) {
    episode_length = 2;
    env = std::make_shared<DelayedRewardEnvironment>(episode_length, state_shape, action_shape, 1.);
  } else if (mode == Action) {
    episode_length = 1;
    env = std::make_shared<ActionRewardEnvironment>(episode_length, state_shape, action_shape);
  } else if (mode == ActionState) {
    episode_length = 1;
    env = std::make_shared<ActionStateRewardEnvironment>(episode_length, state_shape, action_shape);
  }
  num_train_iters = num_explore_iters + num_exploit_iters;
  num_episodes = (num_train_iters + num_eval_iters) / episode_length;

  // set up td3 learning systems
  std::string filename = "configs/" + system + ".yaml";
  torchfort_result_t tstat =
      torchfort_rl_on_policy_create_system("test", filename.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  if (tstat != TORCHFORT_RESULT_SUCCESS) {
    throw std::runtime_error("RL system creation failed");
  }

  // do training loop: initial state
  int iter = 0;
  std::tie(state, reward) = env->initialize();
  for (unsigned int e = 0; e < num_episodes; ++e) {
    done = false;
    int i = 0;
    while (!done) {
      if (iter < num_explore_iters) {
        // explore
        tstat =
            torchfort_rl_on_policy_predict_explore("test",
						   state.data_ptr(), 2, state_batch_shape.data(),
                                                   action.data_ptr(), 2, action_batch_shape.data(),
						   TORCHFORT_FLOAT, 0);
      } else {
        // exploit
        tstat = torchfort_rl_on_policy_predict("test",
					       state.data_ptr(), 2, state_batch_shape.data(),
					       action.data_ptr(), 2, action_batch_shape.data(), TORCHFORT_FLOAT, 0);
      }

      // do environment step
      std::tie(state_new, reward, done) = env->step(action);

      if (iter < num_train_iters) {
        // update replay buffer
        tstat = torchfort_rl_on_policy_update_rollout_buffer("test",
							     state.data_ptr(), 1, state_shape.data(),
                                                             action.data_ptr(), 1, action_shape.data(),
							     &reward, done, TORCHFORT_FLOAT, 0);

        // perform training step if requested:
        tstat = torchfort_rl_on_policy_is_ready("test", is_ready);
        // iterate till there are no more samples inside the buffer:
        if (is_ready) {
          for (unsigned int k = 0; k < num_grad_steps; ++k) {
            tstat = torchfort_rl_on_policy_train_step("test", &p_loss, &q_loss, 0);
          }
          tstat = torchfort_rl_on_policy_reset_rollout_buffer("test");
        }
      }

      // evaluate policy:
      tstat = torchfort_rl_on_policy_evaluate("test",
					      state.data_ptr(), 2, state_batch_shape.data(),
					      action.data_ptr(), 2, action_batch_shape.data(),
					      &q_estimate, 1, reward_batch_shape.data(),
                                              TORCHFORT_FLOAT, 0);

      if (iter >= num_train_iters) {
        auto q_expected = env->spotValue(-1., 1., 0.95);
        qdiff += std::abs(q_expected - q_estimate);
        running_reward += reward;
      }

      if (verbose) {
        std::cout << "episode : " << e << " step: " << i << " state: " << state.item<float>()
                  << " action: " << action.item<float>() << " reward: " << reward << " q: " << q_estimate
                  << " done: " << done << std::endl;
      }

      // copy tensors
      state.copy_(state_new);

      // increase counter:
      i++;
      iter++;
    }
  }

  // compute averages:
  qdiff /= float(num_eval_iters);
  running_reward /= float(num_eval_iters);

  if (verbose) {
    std::cout << "Q-difference: " << qdiff << " average reward: " << running_reward
              << " (expected: " << env->expectedReward(-1., 1.) << ")" << std::endl;
  }

  // do evaluation
  std::tuple<float, float, float> result;
  if (mode == Constant) {
    // the test is successful if reward is predicted correctly
    result = std::make_tuple(qdiff, 0., 1e-2);
  } else if (mode == Predictable) {
    // the test is successful if reward is predicted correctly
    result = std::make_tuple(qdiff, 0., 1e-2);
  } else if (mode == Delayed) {
    // the test is successful if reward is predicted correctly
    result = std::make_tuple(qdiff, 0., 1e-1);
  } else if (mode == Action) {
    // here we just check whether we achieve good reward
    result = std::make_tuple(running_reward, 1., 1e-2);
  } else if (mode == ActionState) {
    // here we also expect great reward
    result = std::make_tuple(running_reward, 1., 1e-2);
  }

  return result;
}

/******************************************************************/
/****************************** PPO *******************************/
/******************************************************************/

TEST(PPO, ConstantEnv) {
  float val, cmp, tol;
  std::tie(val, cmp, tol) = TestSystem(Constant, "ppo", 50000, 0, 100, 8, false);
  EXPECT_NEAR(val, cmp, tol);
}

TEST(PPO, PredictableEnv) {
  float val, cmp, tol;
  std::tie(val, cmp, tol) = TestSystem(Predictable, "ppo", 70000, 0, 100, 8, false);
  EXPECT_NEAR(val, cmp, tol);
}

TEST(PPO, DelayedEnv) {
  float val, cmp, tol;
  std::tie(val, cmp, tol) = TestSystem(Delayed, "ppo", 70000, 0, 100, 8, false);
  EXPECT_NEAR(val, cmp, tol);
}

TEST(PPO, ActionEnv) {
  float val, cmp, tol;
  std::tie(val, cmp, tol) = TestSystem(Action, "ppo", 40000, 1000, 100, 8, false);
  EXPECT_NEAR(val, cmp, 0.3);
}

TEST(PPO, ActionStateEnv) {
  float val, cmp, tol;
  std::tie(val, cmp, tol) = TestSystem(ActionState, "ppo", 40000, 0, 100, 8, false);
  EXPECT_NEAR(val, cmp, 0.3);
}

int main(int argc, char* argv[]) {

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
