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

#pragma once
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
