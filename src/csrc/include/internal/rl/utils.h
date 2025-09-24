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

#pragma once
#include <unordered_map>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/logging.h"
#include "internal/lr_schedulers.h"
#include "internal/model_pack.h"
#include "internal/rl/rl.h"
#include "internal/setup.h"

namespace torchfort {

namespace rl {

// helpers for sanitizing devices
// check if both devices are different, one device has to be a cpu device.
bool validate_devices(int device1, int device2);

// helpers for extracting LRS from optimizer
std::vector<double> get_current_lrs(std::shared_ptr<torch::optim::Optimizer> optimizer);

// helpers for manipulating weights and grads
void init_parameters(std::shared_ptr<ModelWrapper> model);
void copy_parameters(std::shared_ptr<ModelWrapper> target, std::shared_ptr<ModelWrapper> source);
void set_grad_state(std::shared_ptr<ModelWrapper> model, const bool requires_grad);

// polyak update for model averaging:
// computes: target = rho * target + (1-rho) * src
// note that target here denotes the updated model parameters, and src the previous ones
template <typename T>
void polyak_update(std::shared_ptr<ModelWrapper> target, std::shared_ptr<ModelWrapper> source, const T rho) {

  // add no grad guard
  torch::NoGradGuard no_grad;

  // get models
  auto tar = target->parameters();
  auto src = source->parameters();

  // some simple asserts here
  assert(tar.size() == src.size());

  // do in-place update: I don't know a good way of doing that with std::transform:
  for (size_t i = 0; i < tar.size(); ++i) {
    const auto& t = tar[i];
    const auto& s = src[i];
    t.mul_(rho);
    t.add_((1. - rho) * s);
    // t.copy_(torch::Tensor(rho * t + (1.-rho) * s));
  }

  return;
}

// Rescale the action from [a_low, a_high] to [-1, 1]
template <typename T> torch::Tensor scale_action(torch::Tensor unscaled_action, const T& a_low, const T& a_high) {
  auto scaled_action = static_cast<T>(2.0) * ((unscaled_action - a_low) / (a_high - a_low)) - static_cast<T>(1.0);
  scaled_action.to(unscaled_action.dtype());

  return scaled_action;
}

// Unscale the action from [-1., 1.] to [a_low, a_high]
template <typename T> torch::Tensor unscale_action(torch::Tensor scaled_action, const T& a_low, const T& a_high) {
  auto unscaled_action = 0.5 * (a_high - a_low) * (scaled_action + static_cast<T>(1.)) + a_low;
  unscaled_action.to(scaled_action.dtype());

  return unscaled_action;
}

// explained variance
torch::Tensor explained_variance(torch::Tensor q_pred, torch::Tensor q_true, std::shared_ptr<Comm> comm);

} // namespace rl

} // namespace torchfort
