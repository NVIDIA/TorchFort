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
#include "internal/rl/off_policy/ddpg.h"
#include "torchfort.h"

namespace torchfort {

namespace rl {

namespace off_policy {

DDPGSystem::DDPGSystem(const char* name, const YAML::Node& system_node, int model_device, int rb_device)
    : RLOffPolicySystem(model_device, rb_device) {

  // get basic parameters first
  auto algo_node = system_node["algorithm"];
  if (algo_node["parameters"]) {
    auto params = get_params(algo_node["parameters"]);
    std::set<std::string> supported_params{"batch_size", "nstep", "nstep_reward_reduction", "gamma", "rho"};
    check_params(supported_params, params.keys());
    batch_size_ = params.get_param<int>("batch_size")[0];
    gamma_ = params.get_param<float>("gamma")[0];
    rho_ = params.get_param<float>("rho")[0];
    nstep_ = params.get_param<int>("nstep", 1)[0];
    auto redmode = params.get_param<std::string>("nstep_reward_reduction", "sum")[0];
    if (redmode == "sum") {
      nstep_reward_reduction_ = RewardReductionMode::Sum;
    } else if (redmode == "mean") {
      nstep_reward_reduction_ = RewardReductionMode::Mean;
    } else if (redmode == "weighted_mean") {
      nstep_reward_reduction_ = RewardReductionMode::WeightedMean;
    } else if (redmode == "sum_no_skip") {
      nstep_reward_reduction_ = RewardReductionMode::SumNoSkip;
    } else if (redmode == "mean_no_skip") {
      nstep_reward_reduction_ = RewardReductionMode::MeanNoSkip;
    } else if (redmode == "weighted_mean_no_skip") {
      nstep_reward_reduction_ = RewardReductionMode::WeightedMeanNoSkip;
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
      std::set<std::string> supported_params{"a_low", "a_high", "clip",     "sigma_train",     "sigma_explore",
                                             "xi",    "dt",     "adaptive", "noise_actor_type"};
      check_params(supported_params, params.keys());
      a_low_ = params.get_param<float>("a_low")[0];
      a_high_ = params.get_param<float>("a_high")[0];
      float clip = params.get_param<float>("clip")[0];
      float sigma_train = params.get_param<float>("sigma_train")[0];
      float sigma_explore = params.get_param<float>("sigma_explore")[0];
      float mu = 0.f;
      bool adaptive = params.get_param<bool>("adaptive", false)[0];

      // we need to set up the noise actor type:
      if (noise_actor_type == "space_noise") {
        noise_actor_train_ = std::make_shared<ActionNoise<float>>(mu, sigma_train, clip, adaptive);
        noise_actor_exploration_ = std::make_shared<ActionNoise<float>>(mu, sigma_explore, 0.f, adaptive);
      } else if (noise_actor_type == "space_noise_ou") {
        float dt = params.get_param<float>("dt")[0];
        float xi = params.get_param<float>("xi", 0.)[0];
        noise_actor_train_ = std::make_shared<ActionNoiseOU<float>>(mu, sigma_train, clip, dt, xi, adaptive);
        noise_actor_exploration_ = std::make_shared<ActionNoiseOU<float>>(mu, sigma_explore, 0.f, dt, xi, adaptive);
      } else if (noise_actor_type == "parameter_noise") {
        noise_actor_train_ = std::make_shared<ParameterNoise<float>>(mu, sigma_train, clip, adaptive);
        noise_actor_exploration_ = std::make_shared<ParameterNoise<float>>(mu, sigma_explore, 0.f, adaptive);
      } else if (noise_actor_type == "parameter_noise_ou") {
        float dt = params.get_param<float>("dt")[0];
        float xi = params.get_param<float>("xi", 0.)[0];
        noise_actor_train_ = std::make_shared<ParameterNoiseOU<float>>(mu, sigma_train, clip, dt, xi, adaptive);
        noise_actor_exploration_ = std::make_shared<ParameterNoiseOU<float>>(mu, sigma_explore, 0.f, dt, xi, adaptive);
      } else {
        THROW_INVALID_USAGE(noise_actor_type);
      }
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
  // get value model hook
  if (system_node["critic_model"]) {
    q_model_.model = get_model(system_node["critic_model"]);
    q_model_.model->to(model_device_);
    q_model_target_.model = get_model(system_node["critic_model"]);
    q_model_target_.model->to(model_device_);
    // change weights for models
    init_parameters(q_model_.model);
    // copy new parameters
    copy_parameters(q_model_target_.model, q_model_.model);
    // freeze weights for target
    set_grad_state(q_model_target_.model, false);
    // set state
    std::string qname = "critic";
    q_model_.state = get_state(qname.c_str(), system_node);
    qname = "critic_target";
    q_model_target_.state = get_state(qname.c_str(), system_node);
  } else {
    THROW_INVALID_USAGE("Missing critic_model block in configuration file.");
  }

  // get policy model hook
  if (system_node["policy_model"]) {
    p_model_.model = get_model(system_node["policy_model"]);
    p_model_.model->to(model_device_);
    p_model_target_.model = get_model(system_node["policy_model"]);
    p_model_target_.model->to(model_device_);
    // copy parameters
    copy_parameters(p_model_target_.model, p_model_.model);
    // freeze weights
    set_grad_state(p_model_target_.model, false);
    // set state:
    p_model_.state = get_state("actor", system_node);
    p_model_target_.state = get_state("actor_target", system_node);
  } else {
    THROW_INVALID_USAGE("Missing policy_model block in configuration file.");
  }

  // get optimizers
  if (system_node["optimizer"]) {
    // policy model
    p_model_.optimizer = get_optimizer(system_node["optimizer"], p_model_.model);
    // value model
    q_model_.optimizer = get_optimizer(system_node["optimizer"], q_model_.model);
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
    q_model_.lr_scheduler = get_lr_scheduler(system_node["critic_lr_scheduler"], q_model_.optimizer);
  } else {
    THROW_INVALID_USAGE("Missing critic_lr_scheduler block in configuration file.");
  }

  // Setting up general options
  system_state_ = get_state(name, system_node);

  // print settings if requested
  if (system_state_->verbose) {
    printInfo();
  }

  return;
}

void DDPGSystem::printInfo() const {
  std::cout << "DDPG parameters:" << std::endl;
  std::cout << "batch_size = " << batch_size_ << std::endl;
  std::cout << "nstep = " << nstep_ << std::endl;
  std::cout << "reward_nstep_mode = " << nstep_reward_reduction_ << std::endl;
  std::cout << "gamma = " << gamma_ << std::endl;
  std::cout << "rho = " << rho_ << std::endl;
  std::cout << "a_low = " << a_low_ << std::endl;
  std::cout << "a_high = " << a_high_ << std::endl << std::endl;
  std::cout << "training noise:" << std::endl;
  noise_actor_train_->printInfo();
  std::cout << "exploration noise:" << std::endl;
  noise_actor_exploration_->printInfo();
  std::cout << std::endl;
  std::cout << "replay buffer:" << std::endl;
  replay_buffer_->printInfo();
  return;
}

torch::Device DDPGSystem::modelDevice() const { return model_device_; }

torch::Device DDPGSystem::rbDevice() const { return rb_device_; }

int DDPGSystem::getRank() const {
  if (!system_comm_) {
    return 0;
  } else {
    return system_comm_->rank;
  }
}

void DDPGSystem::initSystemComm(MPI_Comm mpi_comm) {
  // Set up distributed communicators for all models
  // system
  system_comm_ = std::make_shared<Comm>(mpi_comm);
  system_comm_->initialize(model_device_.is_cuda());
  // policy
  p_model_.comm = std::make_shared<Comm>(mpi_comm);
  p_model_.comm->initialize(model_device_.is_cuda());
  p_model_target_.comm = std::make_shared<Comm>(mpi_comm);
  p_model_target_.comm->initialize(model_device_.is_cuda());
  // critic
  q_model_.comm = std::make_shared<Comm>(mpi_comm);
  q_model_.comm->initialize(model_device_.is_cuda());
  q_model_target_.comm = std::make_shared<Comm>(mpi_comm);
  q_model_target_.comm->initialize(model_device_.is_cuda());

  // move to device before broadcasting
  // policy
  p_model_.model->to(model_device_);
  p_model_target_.model->to(model_device_);
  // critic
  q_model_.model->to(model_device_);
  q_model_target_.model->to(model_device_);

  // Broadcast initial model parameters from rank 0
  // policy
  for (auto& p : p_model_.model->parameters()) {
    p_model_.comm->broadcast(p, 0);
  }
  for (auto& p : p_model_target_.model->parameters()) {
    p_model_target_.comm->broadcast(p, 0);
  }
  // critic
  for (auto& p : q_model_.model->parameters()) {
    q_model_.comm->broadcast(p, 0);
  }
  for (auto& p : q_model_target_.model->parameters()) {
    q_model_target_.comm->broadcast(p, 0);
  }
}

// Save checkpoint
void DDPGSystem::saveCheckpoint(const std::string& checkpoint_dir) const {
  using namespace torchfort;
  std::filesystem::path root_dir(checkpoint_dir);

  if (!std::filesystem::exists(root_dir)) {
    bool rv = std::filesystem::create_directory(root_dir);
    if (!rv) {
      THROW_INVALID_USAGE("Could not create checkpoint directory " + root_dir.native() + ".");
    }
  }

  // policy
  save_model_pack(p_model_, root_dir / "policy", true);

  // policy target
  save_model_pack(p_model_target_, root_dir / "policy_target", false);

  // critic
  save_model_pack(q_model_, root_dir / "critic", true);

  // critic target
  save_model_pack(q_model_target_, root_dir / "critic_target", false);

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

void DDPGSystem::loadCheckpoint(const std::string& checkpoint_dir) {
  using namespace torchfort;
  std::filesystem::path root_dir(checkpoint_dir);

  // policy
  load_model_pack(p_model_, root_dir / "policy", true);

  // policy target
  load_model_pack(p_model_target_, root_dir / "policy_target", false);

  // critic
  load_model_pack(q_model_, root_dir / "critic", true);

  // critic target
  load_model_pack(q_model_target_, root_dir / "critic_target", false);

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
void DDPGSystem::updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, torch::Tensor r, torch::Tensor d) {
  replay_buffer_->update(s, a, sp, r, d);
}

void DDPGSystem::updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, float r, bool d) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rb_device_);
  torch::Tensor stensu = torch::unsqueeze(s, 0);
  torch::Tensor atensu = torch::unsqueeze(a, 0);
  torch::Tensor sptensu = torch::unsqueeze(sp, 0);
  torch::Tensor rtensu = torch::tensor({r}, options);
  torch::Tensor dtensu = torch::tensor({d ? 1. : 0.}, options);

  DDPGSystem::updateReplayBuffer(stensu, atensu, sptensu, rtensu, dtensu);
}

void DDPGSystem::setSeed(unsigned int seed) { replay_buffer_->setSeed(seed); }

bool DDPGSystem::isReady() { return (replay_buffer_->isReady()); }

std::shared_ptr<ModelState> DDPGSystem::getSystemState_() { return system_state_; }

std::shared_ptr<Comm> DDPGSystem::getSystemComm_() { return system_comm_; }

torch::Tensor DDPGSystem::predictWithNoiseTrain_(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_target_.model->to(model_device_);
  p_model_target_.model->eval();
  state = state.to(model_device_);

  // get noisy prediction
  auto action = (*noise_actor_train_)(p_model_target_, state);

  // clip action
  action = torch::clamp(action, a_low_, a_high_);
  return action;
}

// do exploration step without knowledge about the state
// for example, apply random action
torch::Tensor DDPGSystem::explore(torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // create new empty tensor:
  auto result = torch::empty_like(action).uniform_(a_low_, a_high_);

  return result;
}

torch::Tensor DDPGSystem::predict(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_target_.model->to(model_device_);
  p_model_target_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  auto action = (p_model_target_.model)->forward(std::vector<torch::Tensor>{state})[0];

  // clip action
  action = torch::clamp(action, a_low_, a_high_);

  return action;
}

torch::Tensor DDPGSystem::predictExplore(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_.model->to(model_device_);
  p_model_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  auto action = (*noise_actor_exploration_)(p_model_, state);

  // clip action
  action = torch::clamp(action, a_low_, a_high_);

  return action;
}

torch::Tensor DDPGSystem::evaluate(torch::Tensor state, torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  q_model_target_.model->to(model_device_);
  q_model_target_.model->eval();
  state = state.to(model_device_);
  action = action.to(model_device_);

  // do fwd pass
  torch::Tensor reward = (q_model_target_.model)->forward(std::vector<torch::Tensor>{state, action})[0];

  // squeeze
  reward = torch::squeeze(reward, 1);

  return reward;
}

void DDPGSystem::trainStep(float& p_loss_val, float& q_loss_val) {
  // increment train step counter first, this avoids an initial policy update at start
  train_step_count_++;

  // get samples from replay buffer
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

    // get a new action by predicting one with target network
    ap = predictWithNoiseTrain_(sp);
  }

  // train step
  train_ddpg(p_model_, p_model_target_, q_model_, q_model_target_, s, sp, a, ap, r, d,
             static_cast<float>(std::pow(gamma_, nstep_)), rho_, p_loss_val, q_loss_val);
}

} // namespace off_policy

} // namespace rl

} // namespace torchfort
