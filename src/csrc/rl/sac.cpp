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

#include <algorithm>
#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/rl/distributions.h"
#include "internal/rl/sac.h"

namespace torchfort {

namespace rl {

SACPolicy::SACPolicy(std::shared_ptr<ModelWrapper> p_mu_log_sigma)
    : p_mu_log_sigma_(p_mu_log_sigma), log_sigma_min_(-20.), log_sigma_max_(2.) {}

std::vector<torch::Tensor> SACPolicy::parameters() const {
  std::vector<torch::Tensor> result = p_mu_log_sigma_->parameters();
  return result;
}

void SACPolicy::train() { p_mu_log_sigma_->train(); }

void SACPolicy::eval() { p_mu_log_sigma_->eval(); }

void SACPolicy::to(torch::Device device, bool non_blocking) { p_mu_log_sigma_->to(device, non_blocking); }

void SACPolicy::save(const std::string& fname) const { p_mu_log_sigma_->save(fname); }

void SACPolicy::load(const std::string& fname) { p_mu_log_sigma_->load(fname); }

std::tuple<torch::Tensor, torch::Tensor> SACPolicy::forwardNoise(torch::Tensor state) {
  // predict mu
  auto fwd = p_mu_log_sigma_->forward(std::vector<torch::Tensor>{state});
  auto& action_mu = fwd[0];
  auto& action_log_sigma = fwd[1];
  // predict sigma
  auto action_sigma = torch::exp(torch::clamp(action_log_sigma, log_sigma_min_, log_sigma_max_));

  // create distribution
  auto pi_dist = NormalDistribution(action_mu, action_sigma);

  // sample action and compute log prob
  // do not squash yet
  auto action = pi_dist.rsample();
  auto action_log_prob = torch::sum(torch::flatten(pi_dist.log_prob(action), 1), 1, true);

  // account for squashing
  action_log_prob =
      action_log_prob -
      torch::sum(torch::flatten(2. * (std::log(2.) - action - torch::softplus(-2. * action)), 1), 1, true);
  action = torch::tanh(action);

  return std::make_tuple(action, action_log_prob);
}

torch::Tensor SACPolicy::forwardDeterministic(torch::Tensor state) {
  // predict mu is the only part
  auto action = torch::tanh(p_mu_log_sigma_->forward(std::vector<torch::Tensor>{state})[0]);

  return action;
}

SACSystem::SACSystem(const char* name, const YAML::Node& system_node) : device_(torch::Device(torch::kCPU)) {
  // get device
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));
  device_ = get_device(device_id);

  // get basic parameters first
  auto algo_node = system_node["algorithm"];
  if (algo_node["parameters"]) {
    auto params = get_params(algo_node["parameters"]);
    std::set<std::string> supported_params{"batch_size", "num_critics", "nstep", "nstep_reward_reduction",
                                           "gamma",      "rho",         "alpha"};
    check_params(supported_params, params.keys());
    batch_size_ = params.get_param<int>("batch_size")[0];
    num_critics_ = params.get_param<int>("num_critics", 2)[0];
    gamma_ = params.get_param<float>("gamma")[0];
    rho_ = params.get_param<float>("rho")[0];
    alpha_ = params.get_param<float>("rho")[0];
    nstep_ = params.get_param<int>("nstep", 1)[0];
    auto redmode = params.get_param<std::string>("nstep_reward_reduction", "sum")[0];
    if (redmode == "sum") {
      nstep_reward_reduction_ = RewardReductionMode::Sum;
    } else if (redmode == "mean") {
      nstep_reward_reduction_ = RewardReductionMode::Mean;
    } else if (redmode == "weighted_mean") {
      nstep_reward_reduction_ = RewardReductionMode::WeightedMean;
    } else {
      std::invalid_argument("Unknown nstep_reward_reduction specified");
    }
  } else {
    THROW_INVALID_USAGE("Missing parameters section in algorithm section in configuration file.");
  }

  if (system_node["action"]) {
    auto action_node = system_node["action"];
    std::string noise_actor_type = sanitize(action_node["type"].as<std::string>());
    if (action_node["parameters"]) {
      auto params = get_params(action_node["parameters"]);
      std::set<std::string> supported_params{"a_low", "a_high"};
      check_params(supported_params, params.keys());
      a_low_ = params.get_param<float>("a_low")[0];
      a_high_ = params.get_param<float>("a_high")[0];
    } else {
      THROW_INVALID_USAGE("Missing parameters section in action section in configuration file.");
    }
  } else {
    THROW_INVALID_USAGE("Missing action section in configuration file.");
  }

  if (system_node["replay_buffer"]) {
    auto rb_node = system_node["replay_buffer"];
    std::string rb_type = sanitize(rb_node["type"].as<std::string>());
    if (rb_node["parameters"]) {
      auto params = get_params(rb_node["parameters"]);
      std::set<std::string> supported_params{"type", "max_size", "min_size"};
      check_params(supported_params, params.keys());
      auto max_size = static_cast<size_t>(params.get_param<int>("max_size")[0]);
      auto min_size = static_cast<size_t>(params.get_param<int>("min_size")[0]);

      // distinction between buffer types
      if (rb_type == "uniform") {
        replay_buffer_ = std::make_shared<UniformReplayBuffer>(max_size, min_size);
      } else {
        THROW_INVALID_USAGE(rb_type);
      }
    } else {
      THROW_INVALID_USAGE("Missing parameters section in replay_buffer section in configuration file.");
    }
  } else {
    THROW_INVALID_USAGE("Missing replay_buffer section in configuration file.");
  }

  // general section
  // resize critics vector
  q_models_.resize(num_critics_);
  q_models_target_.resize(num_critics_);

  // get value model hook
  if (system_node["critic_model"]) {
    for (int i = 0; i < q_models_.size(); ++i) {
      q_models_[i].model = get_model(system_node["critic_model"]);
      q_models_target_[i].model = get_model(system_node["critic_model"]);
      // change weights for models
      init_parameters(q_models_[i].model);
      // copy new parameters
      copy_parameters(q_models_target_[i].model, q_models_[i].model);
      // freeze weights for target
      set_grad_state(q_models_target_[i].model, false);
      // set state
      std::string qname = "critic_" + std::to_string(i);
      q_models_[i].state = get_state(qname.c_str(), system_node);
      qname = "critic_target_" + std::to_string(i);
      q_models_target_[i].state = get_state(qname.c_str(), system_node);
    }
  } else {
    THROW_INVALID_USAGE("Missing critic_model block in configuration file.");
  }

  // get policy model hooks
  std::shared_ptr<ModelWrapper> p_model;
  if (system_node["policy_model"]) {
    p_model = get_model(system_node["policy_model"]);
  } else {
    THROW_INVALID_USAGE("Missing policy_model block in configuration file.");
  }
  p_model_.model = std::make_shared<SACPolicy>(std::move(p_model));
  p_model_.state = get_state("actor", system_node);

  // get optimizers
  if (system_node["optimizer"]) {
    // policy model
    p_model_.optimizer = get_optimizer(system_node["optimizer"], p_model_.model->parameters());
    // value model
    for (auto& q_model : q_models_) {
      q_model.optimizer = get_optimizer(system_node["optimizer"], q_model.model);
    }
  } else {
    THROW_INVALID_USAGE("Missing optimizer block in configuration file.");
  }

  // get schedulers
  // policy model
  if (system_node["policy_lr_scheduler"]) {
    p_model_.lr_scheduler = get_lr_scheduler(system_node["policy_lr_scheduler"], p_model_.optimizer);
  } else {
    THROW_INVALID_USAGE("Missing policy_lr_scheduler block in configuration file.");
  }
  // critic models
  if (system_node["critic_lr_scheduler"]) {
    for (auto& q_model : q_models_) {
      q_model.lr_scheduler = get_lr_scheduler(system_node["critic_lr_scheduler"], q_model.optimizer);
    }
  } else {
    THROW_INVALID_USAGE("Missing critic_lr_scheduler block in configuration file.");
  }

  // Setting up general options
  system_state_ = get_state(name, system_node);

  // print settings if requested
  if (system_state_->verbose) {
    printInfo();
  }
}

void SACSystem::printInfo() const {
  std::cout << "SAC parameters:" << std::endl;
  std::cout << "batch_size = " << batch_size_ << std::endl;
  std::cout << "num_critics = " << num_critics_ << std::endl;
  std::cout << "nstep = " << nstep_ << std::endl;
  std::cout << "reward_nstep_mode = " << nstep_reward_reduction_ << std::endl;
  std::cout << "gamma = " << gamma_ << std::endl;
  std::cout << "rho = " << rho_ << std::endl;
  std::cout << "alpha = " << alpha_ << std::endl;
  std::cout << "a_low = " << a_low_ << std::endl;
  std::cout << "a_high = " << a_high_ << std::endl;
  std::cout << std::endl;
  std::cout << "replay buffer:" << std::endl;
  replay_buffer_->printInfo();
  return;
}

void SACSystem::initSystemComm(MPI_Comm mpi_comm) {
  // Set up distributed communicators for all models
  // policy
  p_model_.comm = std::make_shared<Comm>();
  p_model_.comm->initialize(mpi_comm);
  // critic
  for (auto& q_model : q_models_) {
    q_model.comm = std::make_shared<Comm>();
    q_model.comm->initialize(mpi_comm);
  }
  for (auto& q_model_target : q_models_target_) {
    q_model_target.comm = std::make_shared<Comm>();
    q_model_target.comm->initialize(mpi_comm);
  }

  // move to device before broadcasting
  // policy
  p_model_.model->to(device_);
  // critic
  for (auto& q_model : q_models_) {
    q_model.model->to(device_);
  }
  for (auto& q_model_target : q_models_target_) {
    q_model_target.model->to(device_);
  }

  // Broadcast initial model parameters from rank 0
  // policy
  for (auto& p : p_model_.model->parameters()) {
    p_model_.comm->broadcast(p, 0);
  }
  // critic
  for (auto& q_model : q_models_) {
    for (auto& p : q_model.model->parameters()) {
      q_model.comm->broadcast(p, 0);
    }
  }
  for (auto& q_model_target : q_models_target_) {
    for (auto& p : q_model_target.model->parameters()) {
      q_model_target.comm->broadcast(p, 0);
    }
  }

  return;
}

// Save checkpoint
void SACSystem::saveCheckpoint(const std::string& checkpoint_dir) const {
  using namespace torchfort;
  std::filesystem::path root_dir(checkpoint_dir);

  if (!std::filesystem::exists(root_dir)) {
    bool rv = std::filesystem::create_directory(root_dir);
    if (!rv) {
      THROW_INVALID_USAGE("Could not create checkpoint directory.");
    }
  }

  // policy
  {
    auto model_path = root_dir / "policy" / "model.pt";
    p_model_.model->save(model_path.native());

    auto optimizer_path = root_dir / "policy" / "optimizer.pt";
    torch::save(*p_model_.optimizer, optimizer_path.native());

    auto lr_path = root_dir / "policy" / "lr.pt";
    p_model_.lr_scheduler->save(lr_path.native());

    auto state_path = root_dir / "policy" / "state.pt";
    p_model_.state->save(state_path.native());
  }

  // critic
  for (int i = 0; i < q_models_.size(); ++i) {
    auto q_model = q_models_[i];
    std::string subdir = "critic_" + std::to_string(i);
    save_model_pack(q_model, root_dir / subdir, true);
  }

  // critic target
  for (int i = 0; i < q_models_target_.size(); ++i) {
    auto q_model_target = q_models_target_[i];
    std::string subdir = "critic_target_" + std::to_string(i);
    save_model_pack(q_model_target, root_dir / subdir, false);
  }

  // system state
  {
    auto state_path = root_dir / "state.pt";
    system_state_->save(state_path.native());
  }

  // lastly, save the replay buffer:
  {
    auto buffer_path = root_dir / "replay_buffer";
    replay_buffer_->save(buffer_path);
  }
}

void SACSystem::loadCheckpoint(const std::string& checkpoint_dir) {
  using namespace torchfort;
  std::filesystem::path root_dir(checkpoint_dir);

  // policy
  {
    auto model_path = root_dir / "policy" / "model.pt";
    if (!std::filesystem::exists(model_path)) {
      THROW_INVALID_USAGE("Could not find " + model_path.native() + ".");
    }
    p_model_.model->load(model_path.native());

    // connect model and optimizer parameters:
    p_model_.optimizer->parameters() = p_model_.model->parameters();

    auto optimizer_path = root_dir / "policy" / "optimizer.pt";
    if (!std::filesystem::exists(optimizer_path)) {
      THROW_INVALID_USAGE("Could not find " + optimizer_path.native() + ".");
    }
    torch::load(*(p_model_.optimizer), optimizer_path.native());

    auto lr_path = root_dir / "policy" / "lr.pt";
    if (!std::filesystem::exists(lr_path)) {
      THROW_INVALID_USAGE("Could not find " + lr_path.native() + ".");
    }
    p_model_.lr_scheduler->load(lr_path.native(), *(p_model_.optimizer));

    auto state_path = root_dir / "policy" / "state.pt";
    if (!std::filesystem::exists(state_path)) {
      THROW_INVALID_USAGE("Could not find " + state_path.native() + ".");
    }
    p_model_.state->load(state_path.native());
  }

  // critic
  for (int i = 0; i < q_models_.size(); ++i) {
    auto q_model = q_models_[i];
    std::string subdir = "critic_" + std::to_string(i);
    load_model_pack(q_model, root_dir / subdir, true);
  }

  // critic target
  for (int i = 0; i < q_models_target_.size(); ++i) {
    auto q_model_target = q_models_target_[i];
    std::string subdir = "critic_target_" + std::to_string(i);
    load_model_pack(q_model_target, root_dir / subdir, false);
  }

  // system state
  {
    auto state_path = root_dir / "state.pt";
    if (!std::filesystem::exists(state_path)) {
      THROW_INVALID_USAGE("Could not find " + state_path.native() + ".");
    }
    system_state_->load(state_path.native());
  }

  // lastly, load the replay buffer:
  {
    auto buffer_path = root_dir / "replay_buffer";
    if (!std::filesystem::exists(buffer_path)) {
      THROW_INVALID_USAGE("Could not find " + buffer_path.native() + ".");
    }
    replay_buffer_->load(buffer_path);
  }
}

// we should pass a tuple (s, a, s', r, d)
void SACSystem::updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, float r, bool d) {
  // note that we have to rescale the action: [a_low, a_high] -> [-1, 1]
  auto as = scale_action(a, a_low_, a_high_);

  // the replay buffer only stores scaled actions!
  replay_buffer_->update(s, as, sp, r, d);
}

bool SACSystem::isReady() { return (replay_buffer_->isReady()); }

std::shared_ptr<ModelState> SACSystem::getSystemState_() { return system_state_; }

std::shared_ptr<Comm> SACSystem::getSystemComm_() { return system_comm_; }

// do exploration step without knowledge about the state
// for example, apply random action
torch::Tensor SACSystem::explore(torch::Tensor action) {
  // stream
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));

  // no grad guard
  torch::NoGradGuard no_grad;

  // create new empty tensor:
  auto result = torch::empty_like(action).uniform_(a_low_, a_high_);

  return result;
}

torch::Tensor SACSystem::predict(torch::Tensor state) {
  // stream
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));

  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_.model->to(device_);
  p_model_.model->eval();
  state.to(device_);

  // do fwd pass
  auto action = (p_model_.model)->forwardDeterministic(state);

  // clip action
  action = unscale_action(action, a_low_, a_high_);

  return action;
}

torch::Tensor SACSystem::predictExplore(torch::Tensor state) {
  // stream
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));

  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_.model->to(device_);
  p_model_.model->eval();
  state.to(device_);

  // do fwd pass
  torch::Tensor action, log_probs;
  std::tie(action, log_probs) = (p_model_.model)->forwardNoise(state);

  // clip action
  action = unscale_action(action, a_low_, a_high_);

  return action;
}

torch::Tensor SACSystem::evaluate(torch::Tensor state, torch::Tensor action) {
  // stream
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));

  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  q_models_target_[0].model->to(device_);
  q_models_target_[0].model->eval();
  state.to(device_);
  action.to(device_);

  // scale action
  auto action_scale = scale_action(action, a_low_, a_high_);

  // do fwd pass
  auto reward = (q_models_target_[0].model)->forward(std::vector<torch::Tensor>{state, action})[0];

  return reward;
}

void SACSystem::trainStep(float& p_loss_val, float& q_loss_val) {

  // stream
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));

  // increment train step counter first, this avoids an initial policy update at start
  train_step_count_++;

  // we need these
  torch::Tensor s, a, ap, sp, r, d;
  {
    torch::NoGradGuard no_grad;

    // get a sample from the replay buffer
    std::tie(s, a, sp, r, d) = replay_buffer_->sample(batch_size_, gamma_, nstep_, nstep_reward_reduction_);

    // upload to device
    s.to(device_);
    a.to(device_);
    sp.to(device_);
    r.to(device_);
    d.to(device_);
  }

  // train step
  std::vector<float> q_loss_vals;
  train_sac(p_model_, q_models_, q_models_target_, s, sp, a, r, d, static_cast<float>(std::pow(gamma_, nstep_)), rho_,
            alpha_, p_loss_val, q_loss_vals);

  // compute average of q_loss_vals:
  q_loss_val = std::accumulate(q_loss_vals.begin(), q_loss_vals.end(), decltype(q_loss_vals)::value_type(0)) /
               float(q_loss_vals.size());
}

} // namespace rl

} // namespace torchfort
