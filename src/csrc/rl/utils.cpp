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

#include <algorithm>
#include <torch/torch.h>

#include "internal/rl/utils.h"

namespace torchfort {

namespace rl {

bool validate_devices(int device1, int device2) {
  if ((device1 != device2) && (device1 * device2 > 0)) {
    return false;
  } else {
    return true;
  }
}

std::vector<double> get_current_lrs(std::shared_ptr<torch::optim::Optimizer> optimizer) {
  std::vector<double> learnings_rates(optimizer->param_groups().size());
  if (learnings_rates.size() > 0) {
    for (const auto i : c10::irange(optimizer->param_groups().size())) {
      learnings_rates[i] = optimizer->param_groups()[i].options().get_lr();
    }
  }
  return learnings_rates;
}

void init_parameters(std::shared_ptr<ModelWrapper> model) {

  // no grad guard
  torch::NoGradGuard no_grad;

  for (const auto& p : model->named_parameters()) {
    std::string key = p.key();
    auto val = p.value();
    auto dim = val.dim();

    if (key.find("weight") != std::string::npos) {
      // likely a conv or linear, we can use ortho
      if (dim >= 2) {
        torch::nn::init::orthogonal_(val, sqrt(2.));
      } else {
        // likely normalization layer stuff: constant init
        torch::nn::init::constant_(val, 1.);
      }
    } else if (key.find("bias") != std::string::npos) {
      torch::nn::init::constant_(val, 0.);
    }
  }
  return;
}

void copy_parameters(std::shared_ptr<ModelWrapper> target, std::shared_ptr<ModelWrapper> source) {

  // create handles
  auto ptar = target->parameters();
  auto psrc = source->parameters();

  // sanity checks
  assert(ptar.size() == psrc.size());

  // important, apply no grad
  torch::NoGradGuard no_grad;

  // copy loop
  for (size_t i = 0; i < ptar.size(); ++i) {
    auto& t = ptar[i];
    auto& s = psrc[i];
    t.copy_(s);
  }

  return;
}

void set_grad_state(std::shared_ptr<ModelWrapper> model, const bool requires_grad) {
  // create handle for parameters
  auto pars = model->parameters();

  // set grad state
  for (const auto& par : pars) {
    par.requires_grad_(requires_grad);
  }
  return;
}

torch::Tensor explained_variance(torch::Tensor q_pred, torch::Tensor q_true, std::shared_ptr<Comm> comm) {
  // Computes fraction of variance that ypred explains about y.
  // Returns 1 - Var[y-ypred] / Var[y]
  // see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py
  // the communicator is required for distributed training
  torch::Tensor result;
  if (!comm) {
    torch::Tensor var_q_true = torch::var(q_true);
    result = 1. - torch::var(q_true - q_pred) / var_q_true;
  } else {
    // Compute variance of q_true
    std::vector<torch::Tensor> mean_vec = {torch::mean(q_true)};
    comm->allreduce(mean_vec, true);
    torch::Tensor var_q_true = torch::mean(torch::square(q_true - mean_vec[0]));
    mean_vec = {var_q_true};
    comm->allreduce(mean_vec, true);
    var_q_true = mean_vec[0];
    // compute variane of difference:
    mean_vec = {torch::mean(q_true - q_pred)};
    comm->allreduce(mean_vec, true);
    torch::Tensor var_q_diff = torch::mean(torch::square(q_true - q_pred - mean_vec[0]));
    mean_vec = {var_q_diff};
    comm->allreduce(mean_vec, true);
    var_q_diff = mean_vec[0];
    result = 1. - var_q_diff / var_q_true;
  }
  return result;
}

} // namespace rl

} // namespace torchfort
