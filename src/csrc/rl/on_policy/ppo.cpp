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

PPOSystem::PPOSystem(const char* name, const YAML::Node& system_node,
		     torchfort_device_t model_device, torchfort_device_t rb_device)
  : model_device_(get_device(model_device)), rb_device_(get_device(rb_device)) {

  // get basic parameters first
  auto algo_node = system_node["algorithm"];
  if (algo_node["parameters"]) {
    auto params = get_params(algo_node["parameters"]);
    std::set<std::string> supported_params{"batch_size", "gamma", "gae_lambda",
					   "epsilon", "target_kl_divergence",
					   "entropy_loss_coefficient", "value_loss_coefficient",
					   "normalize_advantage"};
    check_params(supported_params, params.keys());
    batch_size_ = params.get_param<int>("batch_size")[0];
    gamma_ = params.get_param<float>("gamma")[0];
    gae_lambda_ = params.get_param<float>("gae_lambda")[0];
    target_kl_divergence_ = params.get_param<float>("target_kl_divergence")[0];
    epsilon_ = params.get_param<float>("epsilon")[0];
    entropy_loss_coeff_ = params.get_param<float>("entropy_loss_coefficient")[0];
    value_loss_coeff_ = params.get_param<float>("value_loss_coefficient")[0];
    normalize_advantage_ = params.get_param<bool>("normalize_advantage")[0];
    
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

  if (system_node["rollout_buffer"]) {
    auto rb_node = system_node["rollout_buffer"];
    std::string rb_type = sanitize(rb_node["type"].as<std::string>());
    if (rb_node["parameters"]) {
      auto params = get_params(rb_node["parameters"]);
      std::set<std::string> supported_params{"type", "size"};
      check_params(supported_params, params.keys());
      auto size = static_cast<size_t>(params.get_param<int>("size")[0]);

      // distinction between buffer types
      if (rb_type == "gae_lambda") {
        rollout_buffer_ = std::make_shared<GAELambdaRolloutBuffer>(size, gamma_, gae_lambda_, rb_device);
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
  // get value model hook
  if (system_node["critic_model"]) {
    q_model_.model = get_model(system_node["critic_model"]);
    // change weights for models
    init_parameters(q_model_.model);
    // set state
    std::string qname = "critic";
    q_model_.state = get_state("critic", system_node);
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
  // PPO policies might be squased
  // TODO: add parameter
  p_model_.model = std::make_shared<GaussianACPolicy>(std::move(p_model), /* squashed = */ true);
  p_model_.state = get_state("actor", system_node);

  // get optimizers
  if (system_node["optimizer"]) {
    // policy model
    p_model_.optimizer = get_optimizer(system_node["optimizer"], p_model_.model->parameters());
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
}

void PPOSystem::printInfo() const {
  std::cout << "PPO parameters:" << std::endl;
  std::cout << "batch_size = " << batch_size_ << std::endl;
  std::cout << "epsilon = " << epsilon_ << std::endl;
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

torch::Device PPOSystem::modelDevice() const {
  return model_device_;
}

torch::Device PPOSystem::rbDevice() const {
  return rb_device_;
}
  
void PPOSystem::initSystemComm(MPI_Comm mpi_comm) {
  // Set up distributed communicators for all models
  // policy
  p_model_.comm = std::make_shared<Comm>(mpi_comm);
  p_model_.comm->initialize(model_device_.is_cuda());
  // critic
  q_model_.comm = std::make_shared<Comm>(mpi_comm);
  q_model_.comm->initialize(model_device_.is_cuda());

  // move to device before broadcasting
  // policy
  p_model_.model->to(model_device_);
  // critic
  q_model_.model->to(model_device_);

  // Broadcast initial model parameters from rank 0
  // policy
  for (auto& p : p_model_.model->parameters()) {
    p_model_.comm->broadcast(p, 0);
  }
  // critic
  for (auto& p : q_model_.model->parameters()) {
    q_model_.comm->broadcast(p, 0);
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
  save_model_pack(q_model_, root_dir / "critic", true);

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
  load_model_pack(q_model_, root_dir / "critic", true);

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

// we should pass a tuple (s, a, s', r, d)
void PPOSystem::updateRolloutBuffer(torch::Tensor s, torch::Tensor a, float r, float q, float log_p, bool e) {
  // note that we have to rescale the action: [a_low, a_high] -> [-1, 1]
  auto as = scale_action(a, a_low_, a_high_);

  // the replay buffer only stores scaled actions!
  rollout_buffer_->update(s, as, r, q, log_p, e);
}

// we need to be able to finalize the buffer
void PPOSystem::finalizeRolloutBuffer(float q, bool e) {
  rollout_buffer_->finalize(q, e);
}

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
  p_model_.model->to(model_device_);
  p_model_.model->eval();
  state.to(model_device_);

  // do fwd pass
  auto action = (p_model_.model)->forwardDeterministic(state);

  // clip action
  action = unscale_action(action, a_low_, a_high_);

  return action;
}

torch::Tensor PPOSystem::predictExplore(torch::Tensor state) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  p_model_.model->to(model_device_);
  p_model_.model->eval();
  state.to(model_device_);

  // do fwd pass
  torch::Tensor action, log_probs;
  std::tie(action, log_probs) = (p_model_.model)->forwardNoise(state);

  // clip action
  action = unscale_action(action, a_low_, a_high_);

  return action;
}

torch::Tensor PPOSystem::evaluate(torch::Tensor state, torch::Tensor action) {
  // no grad guard
  torch::NoGradGuard no_grad;

  // prepare inputs
  q_model_.model->to(model_device_);
  q_model_.model->eval();
  state.to(model_device_);
  action.to(model_device_);

  // scale action
  auto action_scale = scale_action(action, a_low_, a_high_);

  // do fwd pass
  auto reward = (q_model_.model)->forward(std::vector<torch::Tensor>{state, action})[0];

  return reward;
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
    s.to(model_device_);
    a.to(model_device_);
    q.to(model_device_);
    logp.to(model_device_);
    adv.to(model_device_);
    ret.to(model_device_);
  }

  // train step
  train_ppo(p_model_, q_model_,
	    s, a, q, logp, adv, ret,
	    epsilon_, entropy_loss_coeff_, value_loss_coeff_, normalize_advantage_,
	    p_loss_val, q_loss_val, current_kl_divergence_, clip_fraction_);
}

} // namespace on_policy
  
} // namespace rl

} // namespace torchfort
