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

#pragma once
#include <random>
#include <torch/torch.h>

// this file contains some pre-defined environments useful for debugging RL systems
class Environment {
public:
  Environment(const Environment&) = delete;

  // default constructor
  Environment(unsigned int num_steps_per_episode) : num_steps_per_episode_(num_steps_per_episode), num_steps_(0) {}

  virtual std::tuple<torch::Tensor, float, bool> step(torch::Tensor) = 0;
  virtual std::tuple<torch::Tensor, float> initialize() = 0;
  virtual float expectedReward(const float&, const float&) = 0;
  virtual float spotValue(const float&, const float&, const float&) = 0;

protected:
  unsigned int num_steps_per_episode_;
  unsigned int num_steps_;
};

// this is the simplest of the environments, it always emits the same reward independent on action and state. This tests
// if the value function can learn constants
class ConstantRewardEnvironment : public Environment, public std::enable_shared_from_this<Environment> {
public:
  ConstantRewardEnvironment(unsigned int num_steps_per_episode, torch::IntArrayRef state_shape,
                            torch::IntArrayRef action_shape, float default_reward)
      : default_reward_(default_reward), state_shape_(state_shape), action_shape_(action_shape),
        Environment(num_steps_per_episode) {}

  std::tuple<torch::Tensor, float, bool> step(torch::Tensor action) {
    torch::NoGradGuard no_grad;
    num_steps_++;
    return std::make_tuple(torch::zeros(state_shape_, torch::kFloat32), default_reward_,
                           (num_steps_ % num_steps_per_episode_ == 0));
  }

  std::tuple<torch::Tensor, float> initialize() {
    return std::make_tuple(torch::zeros(state_shape_, torch::kFloat32), default_reward_);
  }

  float expectedReward(const float& action_min, const float& action_max) { return default_reward_; }

  float spotValue(const float& action_min, const float& action_max, const float& gamma) { return default_reward_; }

private:
  float default_reward_;
  torch::IntArrayRef state_shape_;
  torch::IntArrayRef action_shape_;
};

// this env emits the reward final_reward at the end of the episode but 0 before. Useful to debug reward discounting
class DelayedRewardEnvironment : public Environment, public std::enable_shared_from_this<Environment> {
public:
  DelayedRewardEnvironment(unsigned int num_steps_per_episode, torch::IntArrayRef state_shape,
                           torch::IntArrayRef action_shape, float final_reward)
      : final_reward_(final_reward), state_shape_(state_shape), action_shape_(action_shape),
        Environment(num_steps_per_episode) {}

  std::tuple<torch::Tensor, float, bool> step(torch::Tensor action) {
    torch::NoGradGuard no_grad;
    num_steps_++;
    float reward = ((num_steps_ % num_steps_per_episode_ == 0) ? final_reward_ : 0.);
    return std::make_tuple(torch::zeros(state_shape_, torch::kFloat32), reward,
                           (num_steps_ % num_steps_per_episode_ == 0));
  }

  std::tuple<torch::Tensor, float> initialize() {
    return std::make_tuple(torch::zeros(state_shape_, torch::kFloat32), 0.);
  }

  float expectedReward(const float& action_min, const float& action_max) { return final_reward_; }

  float spotValue(const float& action_min, const float& action_max, const float& gamma) {
    unsigned int steps_remaining = (num_steps_ % num_steps_per_episode_);
    return std::pow(gamma, steps_remaining) * final_reward_;
  }

private:
  float final_reward_;
  torch::IntArrayRef state_shape_;
  torch::IntArrayRef action_shape_;
};

// this env emits a random reward -1 or +1 and the same value for the state, so it should be easy to predict for the
// value functions. It tests if the network can learn state dependent rewards
class PredictableRewardEnvironment : public Environment, public std::enable_shared_from_this<Environment> {
public:
  PredictableRewardEnvironment(unsigned int num_steps_per_episode, torch::IntArrayRef state_shape,
                               torch::IntArrayRef action_shape)
      : state_shape_(state_shape), action_shape_(action_shape), udist_(0, 1), Environment(num_steps_per_episode) {

    std::random_device dev;
    rngptr_ = std::make_shared<std::mt19937>(dev());
    reward_ = 2 * static_cast<float>(udist_(*rngptr_)) - 1.;
    state_ = torch::empty(state_shape_, torch::kFloat32);
    state_.fill_(reward_);
  }

  std::tuple<torch::Tensor, float> initialize() { return std::make_tuple(state_.clone(), 1.); }

  std::tuple<torch::Tensor, float, bool> step(torch::Tensor action) {
    torch::NoGradGuard no_grad;
    num_steps_++;

    // compute next reward: backup old reward before
    reward_prev_ = reward_;
    reward_ = 2 * static_cast<float>(udist_(*rngptr_)) - 1.;
    state_.fill_(reward_);
    return std::make_tuple(state_.clone(), reward_prev_, (num_steps_ % num_steps_per_episode_ == 0));
  }

  float expectedReward(const float& action_min, const float& action_max) { return 0.; }

  float spotValue(const float& action_min, const float& action_max, const float& gamma) { return reward_prev_; }

private:
  std::shared_ptr<std::mt19937> rngptr_;
  std::uniform_int_distribution<std::mt19937::result_type> udist_;
  torch::IntArrayRef state_shape_;
  torch::IntArrayRef action_shape_;
  torch::Tensor state_;
  float reward_, reward_prev_;
};

// this env produces a reward proprotional to the mean value of the action: useful to test if the
// policy can learn to maximize the action values
class ActionRewardEnvironment : public Environment, public std::enable_shared_from_this<Environment> {
public:
  ActionRewardEnvironment(unsigned int num_steps_per_episode, torch::IntArrayRef state_shape,
                          torch::IntArrayRef action_shape)
      : state_shape_(state_shape), action_shape_(action_shape), Environment(num_steps_per_episode) {}

  std::tuple<torch::Tensor, float> initialize() {
    return std::make_tuple(torch::zeros(state_shape_, torch::kFloat32), 0.);
  }

  std::tuple<torch::Tensor, float, bool> step(torch::Tensor action) {
    torch::NoGradGuard no_grad;
    num_steps_++;

    // extract new reward from action:
    float reward = torch::mean(action).item<float>();
    bool done = (num_steps_ % num_steps_per_episode_ == 0);

    // compute next reward
    return std::make_tuple(torch::zeros(state_shape_, torch::kFloat32), reward, done);
  }

  float expectedReward(const float& action_min, const float& action_max) { return action_max; }

  float spotValue(const float& action_min, const float& action_max, const float& gamma) { return action_max; }

private:
  torch::IntArrayRef state_shape_;
  torch::IntArrayRef action_shape_;
};

// this env returns ation * state rewards and tests the interconnection between value and policy networks
class ActionStateRewardEnvironment : public Environment, public std::enable_shared_from_this<Environment> {
public:
  ActionStateRewardEnvironment(unsigned int num_steps_per_episode, torch::IntArrayRef state_shape,
                               torch::IntArrayRef action_shape)
      : state_shape_(state_shape), action_shape_(action_shape), udist_(0, 1), Environment(num_steps_per_episode) {

    std::random_device dev;
    rngptr_ = std::make_shared<std::mt19937>(dev());
    state_val_ = 2 * static_cast<float>(udist_(*rngptr_)) - 1.;
    state_ = torch::empty(state_shape_, torch::kFloat32);
    state_.fill_(state_val_);
  }

  std::tuple<torch::Tensor, float> initialize() { return std::make_tuple(state_.clone(), 1.); }

  std::tuple<torch::Tensor, float, bool> step(torch::Tensor action) {
    torch::NoGradGuard no_grad;
    num_steps_++;

    // backup the current reward
    float reward_fact = torch::mean(action).item<float>();
    float reward = reward_fact * state_val_;

    // compute next reward, backup old state value before
    state_val_prev_ = state_val_;
    state_val_ = 2 * static_cast<float>(udist_(*rngptr_)) - 1.;
    state_.fill_(state_val_);
    return std::make_tuple(state_.clone(), reward, (num_steps_ % num_steps_per_episode_ == 0));
  }

  float expectedReward(const float& action_min, const float& action_max) {
    return 0.5 * ((-1. * action_min) + (1. * action_max));
  }

  float spotValue(const float& action_min, const float& action_max, const float& gamma) {
    return (state_val_prev_ < 0 ? action_min : action_max) * state_val_prev_;
  }

private:
  std::shared_ptr<std::mt19937> rngptr_;
  std::uniform_int_distribution<std::mt19937::result_type> udist_;
  torch::IntArrayRef state_shape_;
  torch::IntArrayRef action_shape_;
  torch::Tensor state_;
  float state_val_, state_val_prev_;
};
