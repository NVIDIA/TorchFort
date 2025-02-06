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
#include "internal/rl/off_policy/sac.h"

namespace torchfort {

namespace rl {

namespace off_policy {

void AlphaModel::setup(const float& alpha_value) {
  float alpha_val = alpha_value;
  alpha_coeff_ = true;
  if (alpha_val < 0.) {
    THROW_INVALID_USAGE("Invalid entropy coefficient alpha: has to be >= 0.");
  }

  if (alpha_val == 0.) {
    alpha_coeff_ = false;
    // make sure the log does not blow up, we check for alpha_coeff later
    alpha_val = 0.01;
  }
  log_alpha_ = register_parameter("log_alpha",
                                  torch::tensor({std::log(alpha_val)}, torch::TensorOptions().dtype(torch::kFloat32)));
}

// Implement the forward function.
torch::Tensor AlphaModel::forward(torch::Tensor input) {
  torch::Tensor output;
  if (alpha_coeff_) {
    output = torch::exp(log_alpha_) * input;
  } else {
    output = input * 0.;
  }
  return output;
}

SACSystem::SACSystem(const char* name, const YAML::Node& system_node, int model_device, int rb_device)
    : RLOffPolicySystem(model_device, rb_device) {

  // get basic parameters first
  auto algo_node = system_node["algorithm"];
  if (algo_node["parameters"]) {
    auto params = get_params(algo_node["parameters"]);
    std::set<std::string> supported_params{"batch_size", "num_critics", "nstep", "nstep_reward_reduction",
                                           "gamma",      "rho",         "alpha", "target_entropy"};
    check_params(supported_params, params.keys());
    batch_size_ = params.get_param<int>("batch_size")[0];
    num_critics_ = params.get_param<int>("num_critics", 2)[0];
    gamma_ = params.get_param<float>("gamma")[0];
    rho_ = params.get_param<float>("rho")[0];
    // alpha needs special care
    AlphaModel am;
    am.setup(params.get_param<float>("alpha", 0.)[0]);
    alpha_model_ = std::make_shared<AlphaModel>(am);
    target_entropy_ = params.get_param<float>("target_entropy_", 1.)[0];
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

  if (system_node["actor"]) {
    auto actor_node = system_node["actor"];
    std::string noise_actor_type = sanitize(actor_node["type"].as<std::string>());
    if (actor_node["parameters"]) {
      auto params = get_params(actor_node["parameters"]);
      std::set<std::string> supported_params{"a_low", "a_high"};
      check_params(supported_params, params.keys());
      a_low_ = params.get_param<float>("a_low")[0];
      a_high_ = params.get_param<float>("a_high")[0];
    } else {
      THROW_INVALID_USAGE("Missing parameters section in actor section in configuration file.");
    }
  } else {
    THROW_INVALID_USAGE("Missing actor section in configuration file.");
  }

  if (system_node["replay_buffer"]) {
    auto rb_node = system_node["replay_buffer"];
    std::string rb_type = sanitize(rb_node["type"].as<std::string>());
    if (rb_node["parameters"]) {
      auto params = get_params(rb_node["parameters"]);
      std::set<std::string> supported_params{"type", "max_size", "min_size", "n_envs"};
      check_params(supported_params, params.keys());
      auto max_size = static_cast<size_t>(params.get_param<int>("max_size")[0]);
      auto min_size = static_cast<size_t>(params.get_param<int>("min_size")[0]);
      auto n_envs = static_cast<size_t>(params.get_param<int>("n_envs", 1)[0]);

      // distinction between buffer types
      if (rb_type == "uniform") {
        replay_buffer_ = std::make_shared<UniformReplayBuffer>(max_size, min_size, n_envs, gamma_, nstep_,
                                                               nstep_reward_reduction_, rb_device);
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
    auto policy_node = system_node["policy_model"];

    // get basic policy parameters:
    p_model = get_model(policy_node);
  } else {
    THROW_INVALID_USAGE("Missing policy_model block in configuration file.");
  }
  // SAC policies should always be squashed
  p_model_.model = std::make_shared<GaussianPolicy>(std::move(p_model), /* squashed = */ true);
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

  // in this case we want to optimize the entropy coefficient
  if (system_node["alpha_optimizer"]) {
    // register alpha as a new parameter
    alpha_optimizer_ = get_optimizer(system_node["alpha_optimizer"], alpha_model_->parameters());
  } else {
    alpha_optimizer_ = nullptr;
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

  // alpha parameter
  if (system_node["alpha_lr_scheduler"] && alpha_optimizer_) {
    alpha_lr_scheduler_ = get_lr_scheduler(system_node["alpha_lr_scheduler"], alpha_optimizer_);
  } else {
    alpha_lr_scheduler_ = nullptr;
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
  if (alpha_model_->alpha_coeff_) {
    std::cout << "alpha = " << torch::exp(alpha_model_->log_alpha_).item<float>() << std::endl;
  } else {
    std::cout << "alpha = " << 0. << std::endl;
  }
  if (alpha_optimizer_) {
    std::cout << "learnable alpha = " << true << std::endl;
  } else {
    std::cout << "learnable alpha = " << false << std::endl;
  }
  std::cout << "a_low = " << a_low_ << std::endl;
  std::cout << "a_high = " << a_high_ << std::endl;
  std::cout << std::endl;
  std::cout << "replay buffer:" << std::endl;
  replay_buffer_->printInfo();
  return;
}

torch::Device SACSystem::modelDevice() const { return model_device_; }

torch::Device SACSystem::rbDevice() const { return rb_device_; }

int SACSystem::getRank() const	{
  if (!system_comm_) {
    return 0;
  } else {
    return system_comm_->rank;
  }
}

void SACSystem::initSystemComm(MPI_Comm mpi_comm) {
  // Set up distributed communicators for all models
  // system
  system_comm_ = std::make_shared<Comm>(mpi_comm);
  system_comm_->initialize(model_device_.is_cuda());
  // policy
  p_model_.comm = std::make_shared<Comm>(mpi_comm);
  p_model_.comm->initialize(model_device_.is_cuda());
  // critic
  for (auto& q_model : q_models_) {
    q_model.comm = std::make_shared<Comm>(mpi_comm);
    q_model.comm->initialize(model_device_.is_cuda());
  }
  for (auto& q_model_target : q_models_target_) {
    q_model_target.comm = std::make_shared<Comm>(mpi_comm);
    q_model_target.comm->initialize(model_device_.is_cuda());
  }

  // move to device before broadcasting
  // policy
  p_model_.model->to(model_device_);
  // critic
  for (auto& q_model : q_models_) {
    q_model.model->to(model_device_);
  }
  for (auto& q_model_target : q_models_target_) {
    q_model_target.model->to(model_device_);
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
      THROW_INVALID_USAGE("Could not create checkpoint directory " + root_dir.native() + ".");
    }
  }

  // policy
  {
    std::filesystem::path policy_root_dir = root_dir / "policy";
    if (!std::filesystem::exists(policy_root_dir)) {
      bool rv = std::filesystem::create_directory(policy_root_dir);
      if (!rv) {
        THROW_INVALID_USAGE("Could not create policy checkpoint directory " + policy_root_dir.native() + ".");
      }
    }
    auto model_path = policy_root_dir / "model.pt";
    p_model_.model->save(model_path.native());

    auto optimizer_path = policy_root_dir / "optimizer.pt";
    torch::save(*p_model_.optimizer, optimizer_path.native());

    auto lr_path = policy_root_dir / "lr.pt";
    p_model_.lr_scheduler->save(lr_path.native());

    auto state_path = policy_root_dir / "state.pt";
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

  // alpha
  if (alpha_optimizer_) {

    std::filesystem::path alpha_root_dir = root_dir / "alpha";
    if (!std::filesystem::exists(alpha_root_dir)) {
      bool rv = std::filesystem::create_directory(alpha_root_dir);
      if (!rv) {
        THROW_INVALID_USAGE("Could not create alpha checkpoint directory " + alpha_root_dir.native() + ".");
      }
    }

    auto model_path = alpha_root_dir / "model.pt";
    alpha_model_->to(torch::Device(torch::kCPU));
    torch::save(alpha_model_, model_path.native());
    alpha_model_->to(model_device_);

    auto optimizer_path = alpha_root_dir / "optimizer.pt";
    torch::save(*alpha_optimizer_, optimizer_path.native());

    if (alpha_lr_scheduler_) {
      auto lr_path = alpha_root_dir / "lr.pt";
      alpha_lr_scheduler_->save(lr_path.native());
    }
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

  // alpha
  if (alpha_optimizer_) {
    auto model_path = root_dir / "alpha" / "model.pt";
    if (!std::filesystem::exists(model_path)) {
      THROW_INVALID_USAGE("Could not find " + model_path.native() + ".");
    }
    alpha_model_->to(torch::Device(torch::kCPU));
    torch::load(alpha_model_, model_path.native());
    alpha_model_->to(model_device_);

    // connect model and optimizer parameters:
    alpha_optimizer_->parameters() = alpha_model_->parameters();

    auto optimizer_path = root_dir / "alpha" / "optimizer.pt";
    if (!std::filesystem::exists(optimizer_path)) {
      THROW_INVALID_USAGE("Could not find " + optimizer_path.native() + ".");
    }
    torch::load(*(alpha_optimizer_), optimizer_path.native());

    if (alpha_lr_scheduler_) {
      auto lr_path = root_dir / "alpha" / "lr.pt";
      if (!std::filesystem::exists(lr_path)) {
        THROW_INVALID_USAGE("Could not find " + lr_path.native() + ".");
      }
      alpha_lr_scheduler_->load(lr_path.native(), *(alpha_optimizer_));
    }
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
void SACSystem::updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, torch::Tensor r, torch::Tensor d) {
  // note that we have to rescale the action: [a_low, a_high] -> [-1, 1],
  // but the replay buffer only stores scaled actions! 
  replay_buffer_->update(s, a, sp, r, d);
}

void SACSystem::updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, float r, bool d) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rb_device_);
  torch::Tensor stensu = torch::unsqueeze(s, 0);
  torch::Tensor atensu = torch::unsqueeze(a, 0);
  torch::Tensor sptensu = torch::unsqueeze(sp, 0);
  torch::Tensor rtensu = torch::tensor({r}, options);
  torch::Tensor dtensu = torch::tensor({d ? 1. : 0.}, options);

  SACSystem::updateReplayBuffer(stensu, atensu, sptensu, rtensu, dtensu);
}

void SACSystem::setSeed(unsigned int seed) { replay_buffer_->setSeed(seed); }

bool SACSystem::isReady() { return (replay_buffer_->isReady()); }

std::shared_ptr<ModelState> SACSystem::getSystemState_() { return system_state_; }

std::shared_ptr<Comm> SACSystem::getSystemComm_() { return system_comm_; }

// do exploration step without knowledge about the state
// for example, apply random action
torch::Tensor SACSystem::explore(torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // create new empty tensor:
  auto result = torch::empty_like(action).uniform_(a_low_, a_high_);

  return result;
}

torch::Tensor SACSystem::predict(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_.model->to(model_device_);
  p_model_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  auto action = (p_model_.model)->forwardDeterministic(state);

  // clip action
  action = unscale_action(action, a_low_, a_high_);

  return action;
}

torch::Tensor SACSystem::predictExplore(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_.model->to(model_device_);
  p_model_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  torch::Tensor action, log_probs;
  std::tie(action, log_probs) = (p_model_.model)->forwardNoise(state);

  // clip action
  action = unscale_action(action, a_low_, a_high_);

  return action;
}

torch::Tensor SACSystem::evaluate(torch::Tensor state, torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  q_models_target_[0].model->to(model_device_);
  q_models_target_[0].model->eval();
  state = state.to(model_device_);
  action = action.to(model_device_);

  // do fwd pass
  torch::Tensor reward = (q_models_target_[0].model)->forward(std::vector<torch::Tensor>{state, action})[0];

  // squeeze
  reward = torch::squeeze(reward, 1);

  return reward;
}

void SACSystem::trainStep(float& p_loss_val, float& q_loss_val) {
  // increment train step counter first, this avoids an initial policy update at start
  train_step_count_++;

  // we need these
  torch::Tensor s, a, ap, sp, r, d;
  {
    torch::NoGradGuard no_grad;

    // get a sample from the replay buffer
    std::tie(s, a, sp, r, d) = replay_buffer_->sample(batch_size_);

    // upload to device
    s = s.to(model_device_);
    a = a.to(model_device_);
    sp = sp.to(model_device_);
    r = r.to(model_device_);
    d = d.to(model_device_);
  }

  // train step
  train_sac(p_model_, q_models_, q_models_target_, s, sp, a, r, d, alpha_model_, alpha_optimizer_, alpha_lr_scheduler_,
            target_entropy_, static_cast<float>(std::pow(gamma_, nstep_)), rho_, p_loss_val, q_loss_val);
}

} // namespace off_policy

} // namespace rl

} // namespace torchfort
