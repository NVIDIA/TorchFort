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
#include <limits>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

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
#include "internal/setup.h"

// rl stuff
#include "internal/rl/distributions.h"
#include "internal/rl/rl.h"

namespace torchfort {

namespace rl {

enum ActorNormalizationMode { Clip = 1, Scale = 2 };

class TanhBijector {
public:
  TanhBijector() {}

  static torch::Tensor forward(torch::Tensor x) { return torch::tanh(x); }

  static torch::Tensor atanh(torch::Tensor x) { return 0.5 * (x.log1p() - (-x).log1p()); }

  static torch::Tensor inverse(torch::Tensor y) {
    auto eps = std::numeric_limits<float>::epsilon();
    // Clip to avoid NaN
    auto yclamp = torch::clamp(y, -1.0 + eps, 1.0 - eps);
    return TanhBijector::atanh(yclamp);
  }

  static torch::Tensor log_prob_correction(torch::Tensor x, float epsilon = 1.e-6) {
    // Squash correction (from original SAC implementation)
    return torch::log(1.0 - torch::square(torch::tanh(x)) + epsilon);
  }
};

// helper for noisy policies
class Policy {
public:
  // default constructor
  Policy() {}
  // disable copy constructor
  Policy(const Policy&) = delete;

  // we expose those for convenience
  virtual std::vector<torch::Tensor> parameters() const = 0;
  virtual void train() = 0;
  virtual void eval() = 0;
  virtual void to(torch::Device device, bool non_blocking = false) = 0;
  virtual void save(const std::string& fname) const = 0;
  virtual void load(const std::string& fname) = 0;
  virtual torch::Device device() const = 0;

  // forward routines
  virtual std::tuple<torch::Tensor, torch::Tensor> evaluateAction(torch::Tensor state, torch::Tensor action) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor> forwardNoise(torch::Tensor state) = 0;
  virtual torch::Tensor forwardDeterministic(torch::Tensor state) = 0;
};

struct PolicyPack {
  std::shared_ptr<Policy> model;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
  std::shared_ptr<BaseLRScheduler> lr_scheduler;
  std::shared_ptr<Comm> comm;
  std::shared_ptr<ModelState> state;
  int grad_accumulation_steps = 1;
};

class GaussianPolicy : public Policy, public std::enable_shared_from_this<Policy> {

public:
  GaussianPolicy(std::shared_ptr<ModelWrapper> p_mu_log_sigma, bool squashed = false);

  // we expose those for convenience
  std::vector<torch::Tensor> parameters() const;
  void train();
  void eval();
  void to(torch::Device device, bool non_blocking = false);
  void save(const std::string& fname) const;
  void load(const std::string& fname);
  torch::Device device() const;

  // forward routines
  std::tuple<torch::Tensor, torch::Tensor> evaluateAction(torch::Tensor state, torch::Tensor action);
  std::tuple<torch::Tensor, torch::Tensor> forwardNoise(torch::Tensor state);
  torch::Tensor forwardDeterministic(torch::Tensor state);

protected:
  std::shared_ptr<NormalDistribution> getDistribution_(torch::Tensor state);

  bool squashed_;
  float log_sigma_min_;
  float log_sigma_max_;
  std::shared_ptr<ModelWrapper> p_mu_log_sigma_;
};

// helper for actor-critic policies
class ACPolicy {
public:
  // default constructor
  ACPolicy() {}
  // disable copy constructor
  ACPolicy(const ACPolicy&) = delete;

  // we expose those for convenience
  virtual std::vector<torch::Tensor> parameters() const = 0;
  virtual void train() = 0;
  virtual void eval() = 0;
  virtual void to(torch::Device device, bool non_blocking = false) = 0;
  virtual void save(const std::string& fname) const = 0;
  virtual void load(const std::string& fname) = 0;
  virtual torch::Device device() const = 0;

  // forward routines
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluateAction(torch::Tensor state,
                                                                                 torch::Tensor action) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forwardNoise(torch::Tensor state) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor> forwardDeterministic(torch::Tensor state) = 0;
};

struct ACPolicyPack {
  std::shared_ptr<ACPolicy> model;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
  std::shared_ptr<BaseLRScheduler> lr_scheduler;
  std::shared_ptr<Comm> comm;
  std::shared_ptr<ModelState> state;
  int grad_accumulation_steps = 1;
};

class GaussianACPolicy : public ACPolicy, public std::enable_shared_from_this<ACPolicy> {

public:
  GaussianACPolicy(std::shared_ptr<ModelWrapper> p_mu_log_sigma_value, bool squashed = false);

  // we expose those for convenience
  std::vector<torch::Tensor> parameters() const;
  void train();
  void eval();
  void to(torch::Device device, bool non_blocking = false);
  void save(const std::string& fname) const;
  void load(const std::string& fname);
  torch::Device device() const;

  // forward routines
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluateAction(torch::Tensor state, torch::Tensor action);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forwardNoise(torch::Tensor state);
  std::tuple<torch::Tensor, torch::Tensor> forwardDeterministic(torch::Tensor state);

protected:
  std::tuple<std::shared_ptr<NormalDistribution>, torch::Tensor> getDistributionValue_(torch::Tensor state);

  bool squashed_;
  float log_sigma_min_;
  float log_sigma_max_;
  std::shared_ptr<ModelWrapper> p_mu_log_sigma_value_;
};

} // namespace rl

} // namespace torchfort
