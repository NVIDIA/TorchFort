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

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

// torchfort
#include "torchfort.h"

// environment
#include "env.h"

// useful for error handling
#define CHECK_TORCHFORT(call)                                                                                          \
  do {                                                                                                                 \
    torchfort_result_t err = call;                                                                                     \
    if (TORCHFORT_RESULT_SUCCESS != err) {                                                                             \
      std::ostringstream os;                                                                                           \
      os << "Error in file " << __FILE__ << ", l" << __LINE__ << ": error code " << err;                               \
      throw std::runtime_error(os.str().c_str());                                                                      \
    }                                                                                                                  \
  } while (false)

int main(int argc, char* argv[]) {

  // load config file
  YAML::Node config = YAML::LoadFile("config_sim.yaml");

  int64_t num_episodes = 10000;
  if (config["num_episodes"]) {
    num_episodes = config["num_episodes"].as<int64_t>();
  }

  int64_t max_steps_per_episode = 500;
  if (config["max_steps_per_episode"]) {
    max_steps_per_episode = config["max_steps_per_episode"].as<int64_t>();
  }

  int eval_frequency = 25;
  if (config["eval_frequency"]) {
    eval_frequency = config["eval_frequency"].as<int>();
  }

  int64_t target_steps = 470;
  if (config["target_steps"]) {
    target_steps = config["target_steps"].as<int64_t>();
  }

  bool normalize_state = false;

  std::cout << "Configuration parameters:" << std::endl;
  std::cout << "Number of episodes: " << num_episodes << std::endl;
  std::cout << "Maximum number of steps per episode: " << max_steps_per_episode << std::endl;
  std::cout << "Evaluation freuency: " << eval_frequency << std::endl;

  // instantiate environment
  CartPoleEnv cenv;
  StateVector state_min, state_max;
  std::tie(state_min, state_max) = cenv.getStateBounds();
  float x_mean = 0.5 * (state_max[0] + state_min[0]);
  float x_width = state_max[0] - state_min[0];
  float theta_mean = 0.5 * (state_max[2] + state_min[2]);
  float theta_width = state_max[2] - state_min[2];

  // instantiate torchfort
#if ENABLE_GPU
  CHECK_TORCHFORT(torchfort_rl_off_policy_create_system("td3_system", "config.yaml", 0, TORCHFORT_DEVICE_CPU));
#else
  CHECK_TORCHFORT(
      torchfort_rl_off_policy_create_system("td3_system", "config.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU));
#endif

  // define variables
  StateVector state, state_new;
  std::array<float, 1> action;
  std::array<int64_t, 2> state_shape_batch = {1, 4};
  std::array<int64_t, 2> action_shape_batch = {1, 1};
  float reward, total_reward;
  bool terminate;

  // allocate cuda arrays
  float *dstate, *dstate_new, *daction, *dreward;
#ifdef ENABLE_GPU
  cudaSetDevice(0);
  cudaMalloc(&dstate, state.size() * sizeof(float));
  cudaMalloc(&dstate_new, state.size() * sizeof(float));
  cudaMalloc(&daction, action.size() * sizeof(float));
#else
  dstate = static_cast<float*>(std::malloc(state.size() * sizeof(float)));
  dstate_new = static_cast<float*>(std::malloc(state.size() * sizeof(float)));
  daction = static_cast<float*>(std::malloc(action.size() * sizeof(float)));
#endif

  int64_t step_total = 0;
  bool is_eval = false;
  bool is_ready = false;
  float ploss, qloss;
  for (int64_t t = 0; t < num_episodes; ++t) {
    // reset the environment and get the initial state:
    state = cenv.reset();

    // normalize state vector
    if (normalize_state) {
      state[0] = (state[0] - x_mean) / x_width;
      state[2] = (state[2] - theta_mean) / theta_width;
    }

    // reset total reward
    total_reward = 0.;

    // test if this is an eval epoch
    is_eval = (t % eval_frequency == 0);
    std::string prefix = (is_eval ? "EVAL::: " : "TRAIN::: ");

    int64_t step;
    for (step = 0; step < max_steps_per_episode; ++step) {
      // increase total step counter
      step_total++;

      // copy data to device
#ifdef ENABLE_GPU
      cudaMemcpy(dstate, state.data(), state.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
      std::memcpy(dstate, state.data(), state.size() * sizeof(float));
#endif

      // state check
      std::cout << prefix + "state: " << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3]
                << std::endl;

      if (is_eval)
        CHECK_TORCHFORT(torchfort_rl_off_policy_predict("td3_system", dstate, 2, state_shape_batch.data(), daction, 2,
                                                        action_shape_batch.data(), TORCHFORT_FLOAT, 0));
      else
        CHECK_TORCHFORT(torchfort_rl_off_policy_predict_explore("td3_system", dstate, 2, state_shape_batch.data(),
                                                                daction, 2, action_shape_batch.data(), TORCHFORT_FLOAT,
                                                                0));

        // copy data to host
#ifdef ENABLE_GPU
      cudaMemcpy(action.data(), daction, action.size() * sizeof(float), cudaMemcpyDeviceToHost);
#else
      std::memcpy(action.data(), daction, action.size() * sizeof(float));
#endif

      // action check
      std::cout << prefix + "action: " << action[0] << std::endl;

      // perform an environment step
      std::tie(state_new, reward, terminate) = cenv.step(action[0]);
      total_reward += reward;

      // normalize state vector
      if (normalize_state) {
        state_new[0] = (state_new[0] - x_mean) / x_width;
        state_new[2] = (state_new[2] - theta_mean) / theta_width;
      }

      // copy data to device
#ifdef ENABLE_GPU
      cudaMemcpy(dstate_new, state_new.data(), state_new.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
      std::memcpy(dstate_new, state_new.data(), state_new.size() * sizeof(float));
#endif

      // update replay buffer
      CHECK_TORCHFORT(torchfort_rl_off_policy_update_replay_buffer(
          "td3_system", dstate, dstate_new, 1, &state_shape_batch[1], daction, 1, &action_shape_batch[1], &reward,
          terminate, TORCHFORT_FLOAT, 0));

      // check if episode is over
      if (terminate) {
        break;
      }

      // update state
      state = state_new;
    }

    // training step
    CHECK_TORCHFORT(torchfort_rl_off_policy_is_ready("td3_system", is_ready));
    if (is_ready && !is_eval) {
      CHECK_TORCHFORT(torchfort_rl_off_policy_train_step("td3_system", &ploss, &qloss, 0));
    }

    // report reward
    std::cout << prefix + "Number of steps till failure: " << step << std::endl;
    std::cout << prefix + "Accumulated episode reward: " << total_reward << std::endl;
    std::cout << prefix + "Mean episode reward: " << total_reward / float(step) << std::endl;

    // log to wandb
    CHECK_TORCHFORT(torchfort_rl_off_policy_wandb_log_float("td3_system", "episode reward", t, total_reward));
    CHECK_TORCHFORT(torchfort_rl_off_policy_wandb_log_float("td3_system", "mean episode reward", t,
                                                            total_reward / float(max_steps_per_episode)));

    // save checkpoint:
    CHECK_TORCHFORT(torchfort_rl_off_policy_save_checkpoint("td3_system", "./checkpoint"));

    // stop if we survived target_steps:
    if (is_eval && (step >= target_steps)) {
      std::cout << prefix + ": training finished: episode length was " << step << " steps >= " << target_steps
                << std::endl;
      break;
    }
  }

  // clean up
#ifdef ENABLE_GPU
  cudaFree(dstate);
  cudaFree(dstate_new);
  cudaFree(daction);
#else
  std::free(dstate);
  std::free(dstate_new);
  std::free(daction);
#endif

  return 0;
}
