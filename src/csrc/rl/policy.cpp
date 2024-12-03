/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/rl/policy.h"

namespace torchfort {

namespace rl {

GaussianPolicy::GaussianPolicy(std::shared_ptr<ModelWrapper> p_mu_log_sigma, bool squashed)
    : squashed_(squashed), p_mu_log_sigma_(p_mu_log_sigma), log_sigma_min_(-20.), log_sigma_max_(2.) {}

std::vector<torch::Tensor> GaussianPolicy::parameters() const {
  std::vector<torch::Tensor> result = p_mu_log_sigma_->parameters();
  return result;
}

void GaussianPolicy::train() { p_mu_log_sigma_->train(); }

void GaussianPolicy::eval() { p_mu_log_sigma_->eval(); }

void GaussianPolicy::to(torch::Device device, bool non_blocking) { p_mu_log_sigma_->to(device, non_blocking); }

void GaussianPolicy::save(const std::string& fname) const { p_mu_log_sigma_->save(fname); }

void GaussianPolicy::load(const std::string& fname) { p_mu_log_sigma_->load(fname); }

torch::Device GaussianPolicy::device() const { return p_mu_log_sigma_->device(); }

std::shared_ptr<NormalDistribution> GaussianPolicy::getDistribution_(torch::Tensor state) {
  // forward step
  auto fwd = p_mu_log_sigma_->forward(std::vector<torch::Tensor>{state});
  // extract mu
  auto& action_mu = fwd[0];
  // extract sigma
  auto& action_log_sigma = fwd[1];
  auto action_sigma = torch::exp(torch::clamp(action_log_sigma, log_sigma_min_, log_sigma_max_));

  // create distribution
  return std::make_shared<NormalDistribution>(action_mu, action_sigma);
}

std::tuple<torch::Tensor, torch::Tensor> GaussianPolicy::evaluateAction(torch::Tensor state, torch::Tensor action) {
  // get distribution
  auto pi_dist = getDistribution_(state);

  // we need to undo squashing in some cases
  torch::Tensor gaussian_action;
  if (squashed_) {
    gaussian_action = torch::atan(action);
  } else {
    gaussian_action = action;
  }

  // compute log prop
  torch::Tensor log_prob = torch::sum(torch::flatten(pi_dist->log_prob(gaussian_action), 1), 1, false);

  // account for squashing
  torch::Tensor entropy;
  if (squashed_) {
    log_prob = log_prob - torch::sum(torch::log(1. - torch::flatten(torch::square(action), 1) + 1.e-6), 1, false);
    // in this case no analytical form for the entropy exists and we need to estimate it from the log probs directly:
    entropy = -log_prob;
  } else {
    // use analytical formula for entropy
    entropy = torch::sum(torch::flatten(pi_dist->entropy(), 1), 1, false);
  }
  return std::make_tuple(log_prob, entropy);
}

std::tuple<torch::Tensor, torch::Tensor> GaussianPolicy::forwardNoise(torch::Tensor state) {
  // get distribution
  auto pi_dist = getDistribution_(state);

  // sample action and compute log prob
  // do not squash yet
  auto action = pi_dist->rsample();
  auto log_prob = torch::sum(torch::flatten(pi_dist->log_prob(action), 1), 1, false);

  // account for squashing
  if (squashed_) {
    log_prob =
        log_prob - torch::sum(torch::flatten(2. * (std::log(2.) - action - torch::softplus(-2. * action)), 1), 1, false);
    action = torch::tanh(action);
  }

  return std::make_tuple(action, log_prob);
}

torch::Tensor GaussianPolicy::forwardDeterministic(torch::Tensor state) {
  // predict mu is the only part
  auto action = p_mu_log_sigma_->forward(std::vector<torch::Tensor>{state})[0];

  if (squashed_) {
    action = torch::tanh(action);
  }

  return action;
}

// combined (actor critic) policies
GaussianACPolicy::GaussianACPolicy(std::shared_ptr<ModelWrapper> p_mu_log_sigma_value, bool squashed)
    : squashed_(squashed), p_mu_log_sigma_value_(p_mu_log_sigma_value), log_sigma_min_(-20.), log_sigma_max_(2.) {}

std::vector<torch::Tensor> GaussianACPolicy::parameters() const {
  std::vector<torch::Tensor> result = p_mu_log_sigma_value_->parameters();
  return result;
}

void GaussianACPolicy::train() { p_mu_log_sigma_value_->train(); }

void GaussianACPolicy::eval() { p_mu_log_sigma_value_->eval(); }

void GaussianACPolicy::to(torch::Device device, bool non_blocking) { p_mu_log_sigma_value_->to(device, non_blocking); }

void GaussianACPolicy::save(const std::string& fname) const { p_mu_log_sigma_value_->save(fname); }

void GaussianACPolicy::load(const std::string& fname) { p_mu_log_sigma_value_->load(fname); }

torch::Device GaussianACPolicy::device() const { return p_mu_log_sigma_value_->device(); }

std::tuple<std::shared_ptr<NormalDistribution>, torch::Tensor>
GaussianACPolicy::getDistributionValue_(torch::Tensor state) {
  // run fwd pass
  auto fwd = p_mu_log_sigma_value_->forward(std::vector<torch::Tensor>{state});
  // extract mu
  auto& action_mu = fwd[0];
  // extract sigma
  auto& action_log_sigma = fwd[1];
  auto action_sigma = torch::exp(torch::clamp(action_log_sigma, log_sigma_min_, log_sigma_max_));
  // extract value
  auto& value = fwd[2];

  // create distribution
  return std::make_tuple(std::make_shared<NormalDistribution>(action_mu, action_sigma), value);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GaussianACPolicy::evaluateAction(torch::Tensor state,
                                                                                         torch::Tensor action) {
  // get distribution and value prediction
  std::shared_ptr<NormalDistribution> pi_dist;
  torch::Tensor value;
  std::tie(pi_dist, value) = getDistributionValue_(state);

  // we need to undo squashing in some cases
  torch::Tensor gaussian_action;
  if (squashed_) {
    gaussian_action = torch::atan(action);
  } else {
    gaussian_action = action;
  }

  // compute log prop
  torch::Tensor log_prob = torch::sum(torch::flatten(pi_dist->log_prob(gaussian_action), 1), 1, false);

  // account for squashing
  torch::Tensor entropy;
  if (squashed_) {
    log_prob = log_prob - torch::sum(torch::log(1. - torch::flatten(torch::square(action), 1) + 1.e-6), 1, false);
    // in this case no analytical form for the entropy exists and we need to estimate it from the log probs directly:
    entropy = -log_prob;
  } else {
    // use analytical formula for entropy
    entropy = torch::sum(torch::flatten(pi_dist->entropy(), 1), 1, false);
  }

  // squeeze value
  value = torch::squeeze(value, 1);
  
  return std::make_tuple(log_prob, entropy, value);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GaussianACPolicy::forwardNoise(torch::Tensor state) {
  // get distribution and value
  std::shared_ptr<NormalDistribution> pi_dist;
  torch::Tensor value;
  std::tie(pi_dist, value) = getDistributionValue_(state);

  // sample action and compute log prob
  // do not squash yet
  auto action = pi_dist->rsample();
  auto log_prob = torch::sum(torch::flatten(pi_dist->log_prob(action), 1), 1, false);

  // account for squashing
  if (squashed_) {
    log_prob =
        log_prob - torch::sum(torch::flatten(2. * (std::log(2.) - action - torch::softplus(-2. * action)), 1), 1, false);
    action = torch::tanh(action);
  }

  // squeeze value
  value = torch::squeeze(value, 1);

  return std::make_tuple(action, log_prob, value);
}

std::tuple<torch::Tensor, torch::Tensor> GaussianACPolicy::forwardDeterministic(torch::Tensor state) {
  // predict mu is the only part
  auto fwd = p_mu_log_sigma_value_->forward(std::vector<torch::Tensor>{state});
  auto action = fwd[0];
  auto value = fwd[2];

  if (squashed_) {
    action = torch::tanh(action);
  }

  // squeeze value
  value = torch::squeeze(value, 1);

  return std::make_tuple(action, value);
}

} // namespace rl

} // namespace torchfort
