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
#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "environments.h"
#include "internal/defines.h"
#include "torchfort.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "test_utils.h"

namespace {

// RAII helper to silence TorchFort's (std::cout) logging during the test bodies. This does not hide
// gtest output, which is written via C stdio rather than std::cout.
struct CoutRedirect {
  CoutRedirect(std::streambuf* new_buffer) : old(std::cout.rdbuf(new_buffer)) {}
  ~CoutRedirect() { std::cout.rdbuf(old); }

private:
  std::streambuf* old;
};

// run a deterministic policy prediction for a fixed scalar state and return the scalar action
float predict_fixed(const std::string& name, float state_value) {
  std::vector<int64_t> state_shape{1}, action_shape{1};
  std::vector<int64_t> state_batch_shape{1, 1}, action_batch_shape{1, 1};
  torch::Tensor state = torch::full(state_shape, state_value, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));
  CHECK_TORCHFORT(torchfort_rl_off_policy_predict(name.c_str(), state.data_ptr(), 2, state_batch_shape.data(),
                                                  action.data_ptr(), 2, action_batch_shape.data(), TORCHFORT_FLOAT, 0));
  return action.item<float>();
}

// fill the replay buffer with random transitions until the system is ready, then perform a few training steps
void train_and_fill(const std::string& name, int num_train_steps) {
  std::vector<int64_t> state_shape{1}, action_shape{1};
  torch::Tensor state = torch::zeros(state_shape, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor state_new = torch::zeros(state_shape, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));

  bool ready = false;
  int pushed = 0;
  const int max_push = 8192; // safety cap, well above any configured min_size
  while (!ready && pushed < max_push) {
    {
      torch::NoGradGuard no_grad;
      state.uniform_(-1., 1.);
      state_new.uniform_(-1., 1.);
      action.uniform_(-1., 1.);
    }
    float reward = 0.5f;
    bool done = (pushed % 8 == 7);
    CHECK_TORCHFORT(torchfort_rl_off_policy_update_replay_buffer(
        name.c_str(), state.data_ptr(), state_new.data_ptr(), 1, state_shape.data(), action.data_ptr(), 1,
        action_shape.data(), &reward, done, TORCHFORT_FLOAT, 0));
    ++pushed;
    CHECK_TORCHFORT(torchfort_rl_off_policy_is_ready(name.c_str(), ready));
  }
  ASSERT_TRUE(ready);

  float p_loss, q_loss;
  for (int i = 0; i < num_train_steps; ++i) {
    CHECK_TORCHFORT(torchfort_rl_off_policy_train_step(name.c_str(), &p_loss, &q_loss, 0));
  }
}

// Round-trip test for both checkpoint restore mechanisms of an off-policy system:
//   * torchfort_rl_off_policy_load_checkpoint -> full restore (weights + replay buffer)
//   * torchfort_rl_off_policy_load_model      -> weights-only restore (e.g. for fine-tuning)
std::string device_suffix(int device) {
  return device == TORCHFORT_DEVICE_CPU ? "cpu" : "gpu" + std::to_string(device);
}

void checkpoint_roundtrip(const std::string& system, int src_device = TORCHFORT_DEVICE_CPU,
                          int restore_device = TORCHFORT_DEVICE_CPU) {
  // silence TorchFort's verbose/logging output for the duration of the test
  std::stringstream cout_buffer;
  CoutRedirect cout_redirect(cout_buffer.rdbuf());

  torch::manual_seed(666);

  const std::string config = get_config_path(system + ".yaml");
  const std::string dev_tag = device_suffix(src_device) + "_to_" + device_suffix(restore_device);
  const std::string src = system + "_" + dev_tag + "_src";
  const std::string full = system + "_" + dev_tag + "_restore_full";
  const std::string weights = system + "_" + dev_tag + "_restore_weights";
  const std::string ckpt_dir = "/tmp/torchfort_rl_ckpt_" + system + "_" + dev_tag;

  std::filesystem::remove_all(ckpt_dir);

  // create and train the source system, filling the replay buffer past its min_size
  CHECK_TORCHFORT(torchfort_rl_off_policy_create_system(src.c_str(), config.c_str(), src_device, src_device));
  train_and_fill(src, /* num_train_steps = */ 5);

  // reference deterministic prediction for a fixed evaluation state
  const float eval_state = 0.5f;
  const float action_ref = predict_fixed(src, eval_state);

  // sanity: the source system has a filled (ready) replay buffer
  bool src_ready = false;
  CHECK_TORCHFORT(torchfort_rl_off_policy_is_ready(src.c_str(), src_ready));
  EXPECT_TRUE(src_ready);

  // save the full checkpoint
  CHECK_TORCHFORT(torchfort_rl_off_policy_save_checkpoint(src.c_str(), ckpt_dir.c_str()));

  // ---------------- full checkpoint restore ----------------
  CHECK_TORCHFORT(
      torchfort_rl_off_policy_create_system(full.c_str(), config.c_str(), restore_device, restore_device));
  // a freshly created system has an empty buffer and is not ready
  bool full_ready_before = true;
  CHECK_TORCHFORT(torchfort_rl_off_policy_is_ready(full.c_str(), full_ready_before));
  EXPECT_FALSE(full_ready_before);

  CHECK_TORCHFORT(torchfort_rl_off_policy_load_checkpoint(full.c_str(), ckpt_dir.c_str()));

  // the policy weights are restored -> identical deterministic prediction
  const float action_full = predict_fixed(full, eval_state);
  EXPECT_NEAR(action_full, action_ref, 1e-5);

  // the replay buffer is restored as well -> the system is ready right away
  bool full_ready_after = false;
  CHECK_TORCHFORT(torchfort_rl_off_policy_is_ready(full.c_str(), full_ready_after));
  EXPECT_TRUE(full_ready_after);

  // training is functional after full checkpoint restore: the policy must change after a few steps.
  // TD3 applies policy gradients every policy_lag (default: 2) critic steps; two steps guarantee
  // at least one policy update regardless of the internal step phase.
  {
    float p_loss, q_loss;
    for (int i = 0; i < 2; ++i) {
      CHECK_TORCHFORT(torchfort_rl_off_policy_train_step(full.c_str(), &p_loss, &q_loss, 0));
    }
  }
  const float action_full_trained = predict_fixed(full, eval_state);
  EXPECT_GT(std::abs(action_full_trained - action_full), 1e-7f);

  // ---------------- weights-only restore (load_model) ----------------
  CHECK_TORCHFORT(torchfort_rl_off_policy_create_system(weights.c_str(), config.c_str(), restore_device,
                                                        restore_device));
  // prediction before loading stems from a fresh (independent) initialization and differs from the reference,
  // which makes the post-load match below a meaningful check that loading actually happened
  const float action_fresh = predict_fixed(weights, eval_state);
  EXPECT_GT(std::abs(action_ref - action_fresh), 1e-6);

  CHECK_TORCHFORT(torchfort_rl_off_policy_load_model(weights.c_str(), ckpt_dir.c_str()));

  // the policy weights are restored -> identical deterministic prediction
  const float action_weights = predict_fixed(weights, eval_state);
  EXPECT_NEAR(action_weights, action_ref, 1e-5);

  // the replay buffer is NOT restored -> the freshly created buffer is still empty and the system is not ready
  bool weights_ready = true;
  CHECK_TORCHFORT(torchfort_rl_off_policy_is_ready(weights.c_str(), weights_ready));
  EXPECT_FALSE(weights_ready);

  // fill the buffer so training is possible, then verify the policy changes after enough steps
  // (two steps guarantee at least one policy update given TD3's policy_lag=2)
  train_and_fill(weights, 0);
  {
    float p_loss, q_loss;
    for (int i = 0; i < 2; ++i) {
      CHECK_TORCHFORT(torchfort_rl_off_policy_train_step(weights.c_str(), &p_loss, &q_loss, 0));
    }
  }
  const float action_weights_trained = predict_fixed(weights, eval_state);
  EXPECT_GT(std::abs(action_weights_trained - action_weights), 1e-7f);

  // cleanup
  std::filesystem::remove_all(ckpt_dir);
}

// ===================================== on-policy (PPO) =====================================

// run a deterministic policy prediction for a fixed scalar state and return the scalar action
float predict_fixed_on_policy(const std::string& name, float state_value) {
  std::vector<int64_t> state_shape{1}, action_shape{1};
  std::vector<int64_t> state_batch_shape{1, 1}, action_batch_shape{1, 1};
  torch::Tensor state = torch::full(state_shape, state_value, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));
  CHECK_TORCHFORT(torchfort_rl_on_policy_predict(name.c_str(), state.data_ptr(), 2, state_batch_shape.data(),
                                                 action.data_ptr(), 2, action_batch_shape.data(), TORCHFORT_FLOAT, 0));
  return action.item<float>();
}

// step a simple environment, pushing transitions into the rollout buffer until the system is ready
// (the rollout buffer becomes ready once it is full and finalized)
void rollout_until_ready(const std::string& name) {
  std::vector<int64_t> state_shape{1}, action_shape{1};
  std::vector<int64_t> state_batch_shape{1, 1}, action_batch_shape{1, 1};
  auto env = std::make_shared<PredictableRewardEnvironment>(1u, state_shape, action_shape);

  torch::Tensor state, state_new, action;
  action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));
  float reward;
  bool done;
  std::tie(state, reward) = env->initialize();

  bool ready = false;
  int guard = 0;
  const int max_iter = 100000; // safety cap, well above any configured rollout buffer size
  while (!ready && guard < max_iter) {
    CHECK_TORCHFORT(torchfort_rl_on_policy_predict_explore(name.c_str(), state.data_ptr(), 2, state_batch_shape.data(),
                                                           action.data_ptr(), 2, action_batch_shape.data(),
                                                           TORCHFORT_FLOAT, 0));
    std::tie(state_new, reward, done) = env->step(action);
    CHECK_TORCHFORT(torchfort_rl_on_policy_update_rollout_buffer(name.c_str(), state.data_ptr(), 1, state_shape.data(),
                                                                 action.data_ptr(), 1, action_shape.data(), &reward,
                                                                 done, TORCHFORT_FLOAT, 0));
    CHECK_TORCHFORT(torchfort_rl_on_policy_is_ready(name.c_str(), ready));
    state = state_new;
    ++guard;
  }
  ASSERT_TRUE(ready);
}

// Round-trip test for both checkpoint restore mechanisms of the on-policy (PPO) system:
//   * torchfort_rl_on_policy_load_checkpoint -> full restore (weights + rollout buffer)
//   * torchfort_rl_on_policy_load_model      -> weights-only restore (e.g. for fine-tuning)
void checkpoint_roundtrip_on_policy(const std::string& system, int src_device = TORCHFORT_DEVICE_CPU,
                                    int restore_device = TORCHFORT_DEVICE_CPU) {
  // silence TorchFort's verbose/logging output for the duration of the test
  std::stringstream cout_buffer;
  CoutRedirect cout_redirect(cout_buffer.rdbuf());

  torch::manual_seed(666);

  const std::string config = get_config_path(system + ".yaml");
  const std::string dev_tag = device_suffix(src_device) + "_to_" + device_suffix(restore_device);
  const std::string src = system + "_" + dev_tag + "_src";
  const std::string full = system + "_" + dev_tag + "_restore_full";
  const std::string weights = system + "_" + dev_tag + "_restore_weights";
  const std::string ckpt_dir = "/tmp/torchfort_rl_ckpt_" + system + "_" + dev_tag;

  std::filesystem::remove_all(ckpt_dir);

  // create the source system and fill its rollout buffer until it is ready
  CHECK_TORCHFORT(torchfort_rl_on_policy_create_system(src.c_str(), config.c_str(), src_device, src_device));
  rollout_until_ready(src);

  // reference deterministic prediction for a fixed evaluation state
  const float eval_state = 0.5f;
  const float action_ref = predict_fixed_on_policy(src, eval_state);

  // sanity: the source system has a filled (ready) rollout buffer
  bool src_ready = false;
  CHECK_TORCHFORT(torchfort_rl_on_policy_is_ready(src.c_str(), src_ready));
  EXPECT_TRUE(src_ready);

  // save the full checkpoint
  CHECK_TORCHFORT(torchfort_rl_on_policy_save_checkpoint(src.c_str(), ckpt_dir.c_str()));

  // ---------------- full checkpoint restore ----------------
  CHECK_TORCHFORT(torchfort_rl_on_policy_create_system(full.c_str(), config.c_str(), restore_device, restore_device));
  // a freshly created system has an empty rollout buffer and is not ready
  bool full_ready_before = true;
  CHECK_TORCHFORT(torchfort_rl_on_policy_is_ready(full.c_str(), full_ready_before));
  EXPECT_FALSE(full_ready_before);

  CHECK_TORCHFORT(torchfort_rl_on_policy_load_checkpoint(full.c_str(), ckpt_dir.c_str()));

  // the network weights are restored -> identical deterministic prediction
  const float action_full = predict_fixed_on_policy(full, eval_state);
  EXPECT_NEAR(action_full, action_ref, 1e-5);

  // the rollout buffer is restored as well -> the system is ready right away
  bool full_ready_after = false;
  CHECK_TORCHFORT(torchfort_rl_on_policy_is_ready(full.c_str(), full_ready_after));
  EXPECT_TRUE(full_ready_after);

  // training is functional after full checkpoint restore: one gradient step must change the policy
  {
    float p_loss, q_loss;
    CHECK_TORCHFORT(torchfort_rl_on_policy_train_step(full.c_str(), &p_loss, &q_loss, 0));
  }
  const float action_full_trained = predict_fixed_on_policy(full, eval_state);
  EXPECT_GT(std::abs(action_full_trained - action_full), 1e-7f);

  // ---------------- weights-only restore (load_model) ----------------
  CHECK_TORCHFORT(
      torchfort_rl_on_policy_create_system(weights.c_str(), config.c_str(), restore_device, restore_device));
  // prediction before loading stems from a fresh (independent) initialization and differs from the reference
  const float action_fresh = predict_fixed_on_policy(weights, eval_state);
  EXPECT_GT(std::abs(action_ref - action_fresh), 1e-6);

  CHECK_TORCHFORT(torchfort_rl_on_policy_load_model(weights.c_str(), ckpt_dir.c_str()));

  // the network weights are restored -> identical deterministic prediction
  const float action_weights = predict_fixed_on_policy(weights, eval_state);
  EXPECT_NEAR(action_weights, action_ref, 1e-5);

  // the rollout buffer is NOT restored -> the freshly created buffer is still empty and the system is not ready
  bool weights_ready = true;
  CHECK_TORCHFORT(torchfort_rl_on_policy_is_ready(weights.c_str(), weights_ready));
  EXPECT_FALSE(weights_ready);

  // fill the rollout buffer so training is possible, then verify one gradient step changes the policy
  rollout_until_ready(weights);
  {
    float p_loss, q_loss;
    CHECK_TORCHFORT(torchfort_rl_on_policy_train_step(weights.c_str(), &p_loss, &q_loss, 0));
  }
  const float action_weights_trained = predict_fixed_on_policy(weights, eval_state);
  EXPECT_GT(std::abs(action_weights_trained - action_weights), 1e-7f);

  // cleanup
  std::filesystem::remove_all(ckpt_dir);
}

} // namespace

TEST(Checkpoint, TD3) { checkpoint_roundtrip("td3"); }
TEST(Checkpoint, DDPG) { checkpoint_roundtrip("ddpg"); }
TEST(Checkpoint, SAC) { checkpoint_roundtrip("sac"); }
TEST(Checkpoint, PPO) { checkpoint_roundtrip_on_policy("ppo"); }

#ifdef ENABLE_GPU
TEST(Checkpoint, TD3GPUtoGPU) { checkpoint_roundtrip("td3", 0, 0); }
TEST(Checkpoint, TD3CPUtoGPU) { checkpoint_roundtrip("td3", TORCHFORT_DEVICE_CPU, 0); }
TEST(Checkpoint, TD3GPUtoCPU) { checkpoint_roundtrip("td3", 0, TORCHFORT_DEVICE_CPU); }

TEST(Checkpoint, DDPGGPUtoGPU) { checkpoint_roundtrip("ddpg", 0, 0); }
TEST(Checkpoint, DDPGCPUtoGPU) { checkpoint_roundtrip("ddpg", TORCHFORT_DEVICE_CPU, 0); }
TEST(Checkpoint, DDPGGPUtoCPU) { checkpoint_roundtrip("ddpg", 0, TORCHFORT_DEVICE_CPU); }

TEST(Checkpoint, SACGPUtoGPU) { checkpoint_roundtrip("sac", 0, 0); }
TEST(Checkpoint, SACCPUtoGPU) { checkpoint_roundtrip("sac", TORCHFORT_DEVICE_CPU, 0); }
TEST(Checkpoint, SACGPUtoCPU) { checkpoint_roundtrip("sac", 0, TORCHFORT_DEVICE_CPU); }

TEST(Checkpoint, PPOGPUtoGPU) { checkpoint_roundtrip_on_policy("ppo", 0, 0); }
TEST(Checkpoint, PPOCPUtoGPU) { checkpoint_roundtrip_on_policy("ppo", TORCHFORT_DEVICE_CPU, 0); }
TEST(Checkpoint, PPOGPUtoCPU) { checkpoint_roundtrip_on_policy("ppo", 0, TORCHFORT_DEVICE_CPU); }
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
