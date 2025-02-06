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

#include <algorithm>
#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/rl/distributions.h"
#include "internal/rl/on_policy/ppo.h"

namespace torchfort {

namespace rl {

namespace on_policy {

PPOSystem::PPOSystem(const char* name, const YAML::Node& system_node, int model_device, int rb_device)
    : RLOnPolicySystem(model_device, rb_device) {

  // get basic parameters first
  auto algo_node = system_node["algorithm"];
  if (algo_node["parameters"]) {
    auto params = get_params(algo_node["parameters"]);
    std::set<std::string> supported_params{"batch_size",
                                           "gamma",
                                           "gae_lambda",
                                           "epsilon",
                                           "clip_q",
                                           "target_kl_divergence",
                                           "entropy_loss_coefficient",
                                           "value_loss_coefficient",
                                           "max_grad_norm",
                                           "normalize_advantage"};
    check_params(supported_params, params.keys());
    batch_size_ = params.get_param<int>("batch_size")[0];
    gamma_ = params.get_param<float>("gamma")[0];
    gae_lambda_ = params.get_param<float>("gae_lambda")[0];
    target_kl_divergence_ = params.get_param<float>("target_kl_divergence")[0];
    epsilon_ = params.get_param<float>("epsilon", 0.2)[0];
    clip_q_ = params.get_param<float>("clip_q", 0.)[0];
    max_grad_norm_ = params.get_param<float>("max_grad_norm", 0.5)[0];
    entropy_loss_coeff_ = params.get_param<float>("entropy_loss_coefficient", 0.0)[0];
    value_loss_coeff_ = params.get_param<float>("value_loss_coefficient", 0.5)[0];
    normalize_advantage_ = params.get_param<bool>("normalize_advantage", true)[0];
  } else {
    THROW_INVALID_USAGE("Missing parameters section in algorithm section in configuration file.");
  }

  std::string noise_actor_type = "gaussian_ac";
  if (system_node["actor"]) {
    auto action_node = system_node["actor"];
    noise_actor_type = sanitize(action_node["type"].as<std::string>());
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
    THROW_INVALID_USAGE("Missing actor section in configuration file.");
  }

  if (system_node["rollout_buffer"]) {
    auto rb_node = system_node["rollout_buffer"];
    std::string rb_type = sanitize(rb_node["type"].as<std::string>());
    if (rb_node["parameters"]) {
      auto params = get_params(rb_node["parameters"]);
      std::set<std::string> supported_params{"type", "size", "n_envs"};
      check_params(supported_params, params.keys());
      auto size = static_cast<size_t>(params.get_param<int>("size")[0]);
      auto n_envs = static_cast<size_t>(params.get_param<int>("n_envs", 1)[0]);

      // distinction between buffer types
      if (rb_type == "gae_lambda") {
        rollout_buffer_ = std::make_shared<GAELambdaRolloutBuffer>(size, n_envs, gamma_, gae_lambda_, rb_device);
      } else {
        THROW_INVALID_USAGE(rb_type);
      }
    } else {
      THROW_INVALID_USAGE("Missing parameters section in rollout_buffer section in configuration file.");
    }
  } else {
    THROW_INVALID_USAGE("Missing rollout_buffer section in configuration file.");
  }

  // general section
  // model
  std::shared_ptr<ModelWrapper> pq_model;
  if (system_node["actor_critic_model"]) {
    auto model_node = system_node["actor_critic_model"];
    pq_model = get_model(model_node);
  } else {
    THROW_INVALID_USAGE("Missing actor_critic_model block in configuration file.");
  }

  // PPO policies might be squased
  // TODO: add parameter
  if (noise_actor_type == "gaussian_ac") {
    pq_model_.model = std::make_shared<GaussianACPolicy>(std::move(pq_model), /* squashed = */ false);
    actor_normalization_mode_ = ActorNormalizationMode::Clip;
  } else if (noise_actor_type == "squashed_gaussian_ac") {
    pq_model_.model = std::make_shared<GaussianACPolicy>(std::move(pq_model), /* squashed = */ true);
    actor_normalization_mode_ = ActorNormalizationMode::Scale;
  } else {
    THROW_INVALID_USAGE(
        "Invalid actor type specified, supported actor types for PPO are [gaussian_ac, squashed_gaussian_ac]");
  }
  pq_model_.state = get_state("actor_critic", system_node);
  pq_model_.model->to(model_device_);

  // get optimizers
  if (system_node["optimizer"]) {
    // policy model
    pq_model_.optimizer = get_optimizer(system_node["optimizer"], pq_model_.model->parameters());
  } else {
    THROW_INVALID_USAGE("Missing optimizer block in configuration file.");
  }

  // get schedulers
  // policy model
  if (system_node["lr_scheduler"]) {
    pq_model_.lr_scheduler = get_lr_scheduler(system_node["lr_scheduler"], pq_model_.optimizer);
  } else {
    THROW_INVALID_USAGE("Missing lr_scheduler block in configuration file.");
  }

  // Setting up general options
  system_state_ = get_state(name, system_node);

  // print settings if requested
  if (system_state_->verbose) {
    printInfo();
  }
}

void PPOSystem::printInfo() const {
  std::cout << "PPO parameters:" << std::endl;
  std::cout << "batch_size = " << batch_size_ << std::endl;
  std::cout << "epsilon = " << epsilon_ << std::endl;
  std::cout << "clip_q = " << clip_q_ << std::endl;
  std::cout << "entropy_loss_coefficient" << entropy_loss_coeff_ << std::endl;
  std::cout << "value_loss_coefficient" << value_loss_coeff_ << std::endl;
  std::cout << "normalize_advantage = " << normalize_advantage_ << std::endl;
  std::cout << "a_low = " << a_low_ << std::endl;
  std::cout << "a_high = " << a_high_ << std::endl;
  std::cout << std::endl;
  std::cout << "rollout buffer:" << std::endl;
  rollout_buffer_->printInfo();
  return;
}

torch::Device PPOSystem::modelDevice() const { return model_device_; }

torch::Device PPOSystem::rbDevice() const { return rb_device_; }

void PPOSystem::initSystemComm(MPI_Comm mpi_comm) {
  // Set up distributed communicators for all models
  // system
  system_comm_ = std::make_shared<Comm>(mpi_comm);
  system_comm_->initialize(model_device_.is_cuda());
  // policy
  pq_model_.comm = std::make_shared<Comm>(mpi_comm);
  pq_model_.comm->initialize(model_device_.is_cuda());

  // move to device before broadcasting
  pq_model_.model->to(model_device_);

  // Broadcast initial model parameters from rank 0
  for (auto& p : pq_model_.model->parameters()) {
    pq_model_.comm->broadcast(p, 0);
  }

  return;
}

// Save checkpoint
void PPOSystem::saveCheckpoint(const std::string& checkpoint_dir) const {
  using namespace torchfort;
  std::filesystem::path root_dir(checkpoint_dir);

  if (!std::filesystem::exists(root_dir)) {
    bool rv = std::filesystem::create_directory(root_dir);
    if (!rv) {
      THROW_INVALID_USAGE("Could not create checkpoint directory " + root_dir.native() + ".");
    }
  }

  // model
  {
    std::filesystem::path model_root_dir = root_dir / "actor_critic";
    if (!std::filesystem::exists(model_root_dir)) {
      bool rv = std::filesystem::create_directory(model_root_dir);
      if (!rv) {
        THROW_INVALID_USAGE("Could not create model checkpoint directory" + model_root_dir.native() + ".");
      }
    }
    auto model_path = model_root_dir / "model.pt";
    pq_model_.model->save(model_path.native());

    auto optimizer_path = model_root_dir / "optimizer.pt";
    torch::save(*pq_model_.optimizer, optimizer_path.native());

    auto lr_path = model_root_dir / "lr.pt";
    pq_model_.lr_scheduler->save(lr_path.native());

    auto state_path = model_root_dir / "state.pt";
    pq_model_.state->save(state_path.native());
  }

  // system state
  {
    auto state_path = root_dir / "state.pt";
    system_state_->save(state_path.native());
  }

  // lastly, save the replay buffer:
  {
    auto buffer_path = root_dir / "rollout_buffer";
    rollout_buffer_->save(buffer_path);
  }
}

void PPOSystem::loadCheckpoint(const std::string& checkpoint_dir) {
  using namespace torchfort;
  std::filesystem::path root_dir(checkpoint_dir);

  // model
  {
    auto model_path = root_dir / "actor_critic" / "model.pt";
    if (!std::filesystem::exists(model_path)) {
      THROW_INVALID_USAGE("Could not find " + model_path.native() + ".");
    }
    pq_model_.model->load(model_path.native());

    // connect model and optimizer parameters:
    pq_model_.optimizer->parameters() = pq_model_.model->parameters();

    auto optimizer_path = root_dir / "actor_critic" / "optimizer.pt";
    if (!std::filesystem::exists(optimizer_path)) {
      THROW_INVALID_USAGE("Could not find " + optimizer_path.native() + ".");
    }
    torch::load(*(pq_model_.optimizer), optimizer_path.native());

    auto lr_path = root_dir / "actor_critic" / "lr.pt";
    if (!std::filesystem::exists(lr_path)) {
      THROW_INVALID_USAGE("Could not find " + lr_path.native() + ".");
    }
    pq_model_.lr_scheduler->load(lr_path.native(), *(pq_model_.optimizer));

    auto state_path = root_dir / "actor_critic" / "state.pt";
    if (!std::filesystem::exists(state_path)) {
      THROW_INVALID_USAGE("Could not find " + state_path.native() + ".");
    }
    pq_model_.state->load(state_path.native());
  }

  // system state
  {
    auto state_path = root_dir / "state.pt";
    if (!std::filesystem::exists(state_path)) {
      THROW_INVALID_USAGE("Could not find " + state_path.native() + ".");
    }
    system_state_->load(state_path.native());
  }

  // lastly, load the rollout buffer:
  {
    auto buffer_path = root_dir / "rollout_buffer";
    if (!std::filesystem::exists(buffer_path)) {
      THROW_INVALID_USAGE("Could not find " + buffer_path.native() + ".");
    }
    rollout_buffer_->load(buffer_path);
  }
}

// convenience function for n_envs=1: if this is not the case, the error will be captured
// in the replay buffer update function, so no need to check it here
void PPOSystem::updateRolloutBuffer(torch::Tensor stens, torch::Tensor atens, float r, bool d) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rb_device_);
  torch::Tensor stensu = torch::unsqueeze(stens, 0);
  torch::Tensor	atensu = torch::unsqueeze(atens, 0);
  torch::Tensor rtens = torch::tensor({r}, options);
  torch::Tensor etens = torch::tensor({d ? 1. : 0.}, options);

  updateRolloutBuffer(stensu, atensu, rtens, etens);
}
  
// we should pass a tuple (s, a, r, d)
void PPOSystem::updateRolloutBuffer(torch::Tensor stens, torch::Tensor atens, torch::Tensor rtens, torch::Tensor etens) {
  // note that we have to rescale the action: [a_low, a_high] -> [-1, 1]
  torch::Tensor as;
  switch (actor_normalization_mode_) {
  case ActorNormalizationMode::Scale:
    // clamp to [a_low, a_high]
    as = torch::clamp(atens, a_low_, a_high_);
    // scale to [-1, 1]
    as = scale_action(as, a_low_, a_high_);
    break;
  case ActorNormalizationMode::Clip:
    // clamp to [a_low, a_high]
    as = torch::clamp(atens, a_low_, a_high_);
    break;
  }

  // compute q:
  torch::Tensor ad = as.to(model_device_);
  torch::Tensor sd = stens.to(model_device_);
  torch::Tensor log_p_tensor, entropy_tensor, value;
  std::tie(log_p_tensor, entropy_tensor, value) = (pq_model_.model)->evaluateAction(sd, ad);

  // the replay buffer only stores scaled actions!
  rollout_buffer_->update(stens, as, rtens, value, log_p_tensor, etens);
}

void PPOSystem::resetRolloutBuffer() { rollout_buffer_->reset(); }

void PPOSystem::setSeed(unsigned int seed) { rollout_buffer_->setSeed(seed); }

bool PPOSystem::isReady() { return (rollout_buffer_->isReady()); }

std::shared_ptr<ModelState> PPOSystem::getSystemState_() { return system_state_; }

std::shared_ptr<Comm> PPOSystem::getSystemComm_() { return system_comm_; }

// do exploration step without knowledge about the state
// for example, apply random action
torch::Tensor PPOSystem::explore(torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // create new empty tensor:
  auto result = torch::empty_like(action).uniform_(a_low_, a_high_);

  return result;
}

torch::Tensor PPOSystem::predict(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  pq_model_.model->to(model_device_);
  pq_model_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  torch::Tensor action, value;
  std::tie(action, value) = (pq_model_.model)->forwardDeterministic(state);

  // clip action
  switch (actor_normalization_mode_) {
  case ActorNormalizationMode::Scale:
    action = unscale_action(action, a_low_, a_high_);
    break;
  case ActorNormalizationMode::Clip:
    action = torch::clamp(action, a_low_, a_high_);
    break;
  }

  return action;
}

torch::Tensor PPOSystem::predictExplore(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  pq_model_.model->to(model_device_);
  pq_model_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  torch::Tensor action, log_probs, value;
  std::tie(action, log_probs, value) = (pq_model_.model)->forwardNoise(state);

  // clip action
  switch (actor_normalization_mode_) {
  case ActorNormalizationMode::Scale:
    action = unscale_action(action, a_low_, a_high_);
    break;
  case ActorNormalizationMode::Clip:
    action = torch::clamp(action, a_low_, a_high_);
    break;
  }

  return action;
}

torch::Tensor PPOSystem::evaluate(torch::Tensor state, torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  pq_model_.model->to(model_device_);
  pq_model_.model->eval();
  state = state.to(model_device_);

  // do fwd pass
  torch::Tensor action_tmp, value;
  std::tie(action_tmp, value) = (pq_model_.model)->forwardDeterministic(state);

  return value;
}

void PPOSystem::trainStep(float& p_loss_val, float& q_loss_val) {
  // increment train step counter first, this avoids an initial policy update at start
  train_step_count_++;

  // we need these
  torch::Tensor s, a, q, logp, adv, ret;
  {
    torch::NoGradGuard no_grad;

    // get a sample from the rollout buffer
    std::tie(s, a, q, logp, adv, ret) = rollout_buffer_->sample(batch_size_);

    // upload to device
    s = s.to(model_device_);
    a = a.to(model_device_);
    q = q.to(model_device_);
    logp = logp.to(model_device_);
    adv = adv.to(model_device_);
    ret = ret.to(model_device_);
  }

  // train step
  train_ppo(pq_model_, s, a, q, logp, adv, ret, epsilon_, clip_q_, entropy_loss_coeff_, value_loss_coeff_,
            max_grad_norm_, target_kl_divergence_, normalize_advantage_, p_loss_val, q_loss_val, current_kl_divergence_,
            clip_fraction_, explained_variance_);

  // system logging
  if ((system_state_->report_frequency > 0) && (train_step_count_ % system_state_->report_frequency == 0)) {
    if (!system_comm_ || (system_comm_ && system_comm_->rank == 0)) {
      std::stringstream os;
      os << "PPO system: "
         << "clip_fraction: " << clip_fraction_ << ", kl_divergence: " << current_kl_divergence_
         << ", explained_variance: " << explained_variance_ << std::endl;
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (system_state_->enable_wandb_hook) {
        torchfort::wandb_log(system_state_, system_comm_, "PPO", "clip_fraction", train_step_count_, clip_fraction_);
        torchfort::wandb_log(system_state_, system_comm_, "PPO", "kl_divergence", train_step_count_,
                             current_kl_divergence_);
        torchfort::wandb_log(system_state_, system_comm_, "PPO", "explained_variance", train_step_count_,
                             explained_variance_);
      }
    }
  }
}

} // namespace on_policy

} // namespace rl

} // namespace torchfort
