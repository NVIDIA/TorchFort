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

#pragma once
#include <unordered_map>

#include <torch/torch.h>

#include "internal/model_pack.h"
#include "internal/rl/rl.h"
#include "internal/utils.h"

namespace torchfort {

namespace rl {

class NoiseActor {

public:
  // remove copy constructor
  NoiseActor(const NoiseActor&) = delete;

  // default constructor
  NoiseActor() {}

  virtual torch::Tensor operator()(const ModelPack&, torch::Tensor) = 0;
  virtual void printInfo() const = 0;
  virtual void freezeNoise(bool) = 0;
};

// classical uncorrelated action space noise
template <typename T> class ActionNoise : public NoiseActor, public std::enable_shared_from_this<NoiseActor> {

public:
  ActionNoise(const T& mu, const T& sigma, const T& clip, const bool& adaptive = false)
      : mu_(mu), sigma_(sigma), clip_(clip), adaptive_(adaptive), initialized_(false), freeze_(false) {}

  torch::Tensor operator()(const ModelPack& policy, torch::Tensor state) override {
    // prepare inputs
    policy.model->to(state.device());

    // get predictions
    auto action = policy.model->forward(std::vector<torch::Tensor>{state})[0];

    T sigma_tmp = sigma_;
    if (adaptive_) {
      sigma_tmp = sigma_ * torch::std(action, /* unbiased = */ false).item<T>();
    }

    // get noise
    if (!initialized_ || !freeze_) {
      noise_state_ = torch::empty_like(action).normal_(mu_, sigma_tmp);
      initialized_ = true;
    }

    torch::Tensor noise;
    if (clip_ > 0) {
      noise = torch::clamp(noise_state_, -clip_, clip_);
    } else {
      noise = noise_state_;
    }

    // apply noise
    action = action + noise;

    return action;
  }

  void printInfo() const {
    std::cout << "action noise parameters:" << std::endl;
    std::cout << "mu = " << mu_ << std::endl;
    std::cout << "sigma = " << sigma_ << std::endl;
    std::cout << "clip = " << clip_ << std::endl;
    std::cout << "adaptive = " << adaptive_ << std::endl;
  }

  void freezeNoise(bool freeze) {
    freeze_ = freeze;
    return;
  }

protected:
  T mu_;
  T sigma_;
  T clip_;
  bool adaptive_;
  bool initialized_;
  bool freeze_;
  torch::Tensor noise_state_;
};

// classical correlated action space noise (OU noise)
// dx_t = xi * (mu - x_t) * dt + sigma * N(0, sqrt(dt))
template <typename T> class ActionNoiseOU : public NoiseActor, public std::enable_shared_from_this<NoiseActor> {

public:
  ActionNoiseOU(const T& mu, const T& sigma, const T& clip, const T& dt, const T& xi, const bool& adaptive = false)
      : mu_(mu), sigma_(sigma), clip_(clip), dt_(dt), xi_(xi), adaptive_(adaptive), initialized_(false),
        freeze_(false) {
    sqrtdt_ = static_cast<T>(std::sqrt(dt_));
  }

  torch::Tensor operator()(const ModelPack& policy, torch::Tensor state) override {
    // prepare inputs
    policy.model->to(state.device());

    // get predictions
    auto action = policy.model->forward(std::vector<torch::Tensor>{state})[0];

    T sigma_tmp = sigma_;
    if (adaptive_) {
      sigma_tmp = sigma_ * torch::std(action, /* unbiased = */ false).item<T>();
    }

    if (!initialized_) {
      // get noise state
      noise_state_ = torch::empty_like(action).normal_(mu_, sigma_tmp);
      initialized_ = true;
    } else if (!freeze_) {
      // update noise state
      torch::Tensor dnoise =
          xi_ * (mu_ - noise_state_) * dt_ + sigma_tmp * torch::empty_like(action).normal_(static_cast<T>(0.), sqrtdt_);
      noise_state_ = noise_state_ + dnoise;
    }

    // apply noise
    torch::Tensor noise;
    if (clip_ > 0) {
      noise = torch::clamp(noise_state_, -clip_, clip_);
    } else {
      noise = noise_state_;
    }

    // add noise to action
    action = action + noise;

    return action;
  }

  void printInfo() const {
    std::cout << "OU action noise parameters:" << std::endl;
    std::cout << "mu = " << mu_ << std::endl;
    std::cout << "sigma = " << sigma_ << std::endl;
    std::cout << "clip = " << clip_ << std::endl;
    std::cout << "dt = " << dt_ << std::endl;
    std::cout << "xi = " << xi_ << std::endl;
  }

  void freezeNoise(bool freeze) {
    freeze_ = freeze;
    return;
  }

protected:
  T mu_;
  T sigma_;
  T clip_;
  T dt_, sqrtdt_;
  T xi_;
  bool initialized_;
  bool freeze_;
  bool adaptive_;
  torch::Tensor noise_state_;
};

// uncorrelated parameter space noise
template <typename T> class ParameterNoise : public NoiseActor, public std::enable_shared_from_this<NoiseActor> {

public:
  ParameterNoise(const T& mu, const T& sigma, const T& clip, const bool& adaptive = false)
      : mu_(mu), sigma_(sigma), clip_(clip), adaptive_(adaptive), initialized_(false), freeze_(false) {}

  torch::Tensor operator()(const ModelPack& policy, torch::Tensor state) override {

    // prepare inputs
    policy.model->to(state.device());

    // backup weights and apply noise:
    auto phandle = policy.model->parameters();
    std::vector<torch::Tensor> parameters;
    {
      // no grad guard:
      torch::NoGradGuard no_grad;

      for (int i = 0; i < phandle.size(); ++i) {

        // create reference
        auto& parameter = phandle[i];

        // backup
        parameters.push_back(parameter.clone());

        // get noise
        if (!initialized_ || !freeze_) {

          // update sigma
          T sigma_tmp = sigma_;
          if (adaptive_) {
            sigma_tmp = sigma_ * torch::std(parameter, /* unbiased = */ false).item<T>();
          }

          if (!initialized_) {
            noise_states_.push_back(torch::empty_like(parameter).normal_(mu_, sigma_tmp));
          } else if (!freeze_) {
            noise_states_[i].normal_(mu_, sigma_tmp);
          }
        }
        // I think we do NOT need to clone here
        auto noise = noise_states_[i];

        // clip if requested
        if (clip_ > 0) {
          noise = torch::clamp(noise, -clip_, clip_);
        }

        // add noise to the parameter
        parameter.copy_(parameter + noise);
      }
      initialized_ = true;
    }

    // get noisy predictions
    auto action = policy.model->forward(std::vector<torch::Tensor>{state})[0];

    // copy back original parameters
    {
      // no grad guard:
      torch::NoGradGuard no_grad;

      for (int i = 0; i < parameters.size(); ++i) {
        phandle[i].copy_(parameters[i]);
      }
    }

    return action;
  }

  void printInfo() const {
    std::cout << "parameter noise parameters:" << std::endl;
    std::cout << "mu = " << mu_ << std::endl;
    std::cout << "sigma = " << sigma_ << std::endl;
    std::cout << "clip = " << clip_ << std::endl;
    std::cout << "adaptive = " << adaptive_ << std::endl;
  }

  void freezeNoise(bool freeze) {
    freeze_ = freeze;
    return;
  }

protected:
  T mu_;
  T sigma_;
  T clip_;
  bool initialized_;
  bool freeze_;
  bool adaptive_;
  std::vector<torch::Tensor> noise_states_;
};

template <typename T> class ParameterNoiseOU : public NoiseActor, public std::enable_shared_from_this<NoiseActor> {

public:
  ParameterNoiseOU(const T& mu, const T& sigma, const T& clip, const T& dt, const T& xi, const bool& adaptive = false)
      : mu_(mu), sigma_(sigma), clip_(clip), dt_(dt), xi_(xi), adaptive_(adaptive), initialized_(false),
        freeze_(false) {
    sqrtdt_ = static_cast<T>(std::sqrt(dt_));
  }

  torch::Tensor operator()(const ModelPack& policy, torch::Tensor state) override {

    // prepare inputs
    policy.model->to(state.device());

    // backup weights and apply noise:
    auto phandle = policy.model->parameters();
    std::vector<torch::Tensor> parameters;
    {
      // no grad guard:
      torch::NoGradGuard no_grad;

      for (int i = 0; i < phandle.size(); ++i) {

        // create reference
        auto& parameter = phandle[i];

        // backup
        parameters.push_back(parameter.clone());

        // get noise
        T sigma_tmp = sigma_;
        if (adaptive_) {
          sigma_tmp = sigma_ * torch::std(parameter, /* unbiased = */ false).item<T>();
        }

        if (!initialized_) {
          // set noise state
          noise_states_.push_back(torch::empty_like(parameter).normal_(mu_, sigma_tmp));
        } else if (!freeze_) {
          // update noise state
          torch::Tensor dnoise = xi_ * (mu_ - noise_states_[i]) * dt_ +
                                 sigma_tmp * torch::empty_like(parameter).normal_(static_cast<T>(0.), sqrtdt_);
          noise_states_[i].add_(dnoise);
        }
        // I think we do NOT need to clone here
        auto noise = noise_states_[i];

        // clip if requested
        if (clip_ > 0) {
          noise = torch::clamp(noise, -clip_, clip_);
        }

        // add noise to the parameter
        parameter.copy_(parameter + noise);
      }
    }

    // check if noise states are initialized:
    if (noise_states_.size() == parameters.size()) {
      initialized_ = true;
    }

    // get noisy predictions
    auto action = policy.model->forward(std::vector<torch::Tensor>{state})[0];

    // copy back original parameters
    {
      // no grad guard:
      torch::NoGradGuard no_grad;

      for (int i = 0; i < parameters.size(); ++i) {
        phandle[i].copy_(parameters[i]);
      }
    }

    return action;
  }

  void printInfo() const {
    std::cout << "OU parameter noise parameters:" << std::endl;
    std::cout << "mu = " << mu_ << std::endl;
    std::cout << "sigma = " << sigma_ << std::endl;
    std::cout << "clip = " << clip_ << std::endl;
    std::cout << "adaptive = " << adaptive_ << std::endl;
  }

  void freezeNoise(bool freeze) {
    freeze_ = freeze;
    return;
  }

protected:
  T mu_;
  T sigma_;
  T clip_;
  T dt_, sqrtdt_;
  T xi_;
  bool initialized_;
  bool freeze_;
  bool adaptive_;
  std::vector<torch::Tensor> noise_states_;
};

} // namespace rl

} // namespace torchfort
