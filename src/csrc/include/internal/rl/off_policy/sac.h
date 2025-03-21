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

#pragma once
#include <unordered_map>

#include <yaml-cpp/yaml.h>

#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/logging.h"
#include "internal/lr_schedulers.h"
#include "internal/model_pack.h"
#include "internal/setup.h"

// rl stuff
#include "internal/rl/off_policy.h"
#include "internal/rl/policy.h"
#include "internal/rl/replay_buffer.h"
#include "internal/rl/utils.h"

namespace torchfort {

namespace rl {

namespace off_policy {

// small helper model for trainable alpha coefficient
struct AlphaModel : torch::nn::Module {

  torch::Tensor forward(torch::Tensor input);
  void setup(const float& alpha_value);

  bool alpha_coeff_;
  torch::Tensor log_alpha_;
};

// implementing https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
template <typename T>
void train_sac(const PolicyPack& p_model, const std::vector<ModelPack>& q_models,
               const std::vector<ModelPack>& q_models_target, torch::Tensor state_old_tensor,
               torch::Tensor state_new_tensor, torch::Tensor action_old_tensor, torch::Tensor reward_tensor,
               torch::Tensor d_tensor, const std::shared_ptr<AlphaModel>& alpha_model,
               const std::shared_ptr<torch::optim::Optimizer>& alpha_optimizer,
               const std::shared_ptr<BaseLRScheduler>& alpha_lr_scheduler, const T& target_entropy, const T& gamma,
               const T& rho, T& p_loss_val, T& q_loss_val) {

  // nvtx marker
  torchfort::nvtx::rangePush("torchfort_train_sac");

  // sanity checks
  // batch size
  auto batch_size = state_old_tensor.size(0);
  assert(batch_size == state_new_tensor.size(0));
  assert(batch_size == action_old_tensor.size(0));
  assert(batch_size == reward_tensor.size(0));
  assert(batch_size == d_tensor.size(0));
  // singleton dims
  assert(reward_tensor.size(1) == 1);
  assert(d_tensor.size(1) == 1);

  // value functions
  // move models to device
  for (const auto& q_model : q_models) {
    q_model.model->train();
  }
  for (const auto& q_model_target : q_models_target) {
    q_model_target.model->train();
  }

  // opt
  // loss is fixed by algorithm
  auto q_loss_func = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));

  // if we are updating the entropy coefficient, do that first
  torch::Tensor alpha_loss;
  if (alpha_optimizer) {
    // compute target entropy
    float targ_ent;
    if (target_entropy > 0.) {
      // we exclude the first dim which is the batch dim
      targ_ent = -float(action_old_tensor.size(1));
      for (int i = 2; i < action_old_tensor.dim(); ++i) {
        targ_ent *= float(action_old_tensor.size(i));
      }
    } else {
      targ_ent = target_entropy;
    }

    // compute action log prob
    torch::Tensor action_tensor, action_log_prob;
    {
      torch::NoGradGuard no_grad;
      std::tie(action_tensor, action_log_prob) = p_model.model->forwardNoise(state_old_tensor);      
      action_log_prob = action_log_prob + targ_ent;
    }
    alpha_loss = -torch::mean(alpha_model->log_alpha_ * action_log_prob);
    alpha_loss.backward();

    // reduce gradients
    if (p_model.comm) {
      std::vector<torch::Tensor> grads;
      grads.reserve(alpha_model->parameters().size());
      for (const auto& p : alpha_model->parameters()) {
        grads.push_back(p.grad());
      }
      p_model.comm->allreduce(grads, true);
    }

    alpha_optimizer->step();
    if (alpha_lr_scheduler) {
      alpha_lr_scheduler->step();
    }
  }

  // policy function
  // compute y: use the target models for q_new, no grads
  torch::Tensor y_tensor;
  {
    torch::NoGradGuard no_grad;
    torch::Tensor action_new_tensor, action_new_log_prob;
    std::tie(action_new_tensor, action_new_log_prob) = p_model.model->forwardNoise(state_new_tensor);

    // compute expected reward
    torch::Tensor q_new_tensor =
      torch::squeeze(q_models_target[0].model->forward(std::vector<torch::Tensor>{state_new_tensor, action_new_tensor})[0], 1);
    for (int i = 1; i < q_models_target.size(); ++i) {
      torch::Tensor q_tmp_tensor =
	torch::squeeze(q_models_target[i].model->forward(std::vector<torch::Tensor>{state_new_tensor, action_new_tensor})[0], 1);
      q_new_tensor = torch::minimum(q_new_tensor, q_tmp_tensor);
    }
    
    // entropy regularization
    q_new_tensor = q_new_tensor - alpha_model->forward(action_new_log_prob);

    // target construction
    y_tensor = torch::Tensor(reward_tensor + q_new_tensor * gamma * (1. - d_tensor));
  }
  
  // backward and update step
  torch::Tensor q_old_tensor =
    torch::squeeze(q_models[0].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_tensor})[0], 1);
  torch::Tensor q_loss_tensor = q_loss_func->forward(q_old_tensor, y_tensor);
  auto state = q_models[0].state;
  if (state->step_train_current % q_models[0].grad_accumulation_steps == 0) {
    q_models[0].optimizer->zero_grad();
  }
  for (int i = 1; i < q_models.size(); ++i) {
    // compute loss
    q_old_tensor = torch::squeeze(q_models[i].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_tensor})[0], 1);
    q_loss_tensor = q_loss_tensor + q_loss_func->forward(q_old_tensor, y_tensor);
    state = q_models[i].state;
    if (state->step_train_current % q_models[i].grad_accumulation_steps == 0) {
      q_models[i].optimizer->zero_grad();
    }
  }
  q_loss_tensor.backward();

  // update critics
  for (const auto& q_model : q_models) {

    // finish gradient accumulation
    state = q_model.state;
    if ((state->step_train_current + 1) % q_model.grad_accumulation_steps == 0) { 
      // grad comm
      if (q_model.comm) {
	std::vector<torch::Tensor> grads;
	grads.reserve(q_model.model->parameters().size());
	for (const auto& p : q_model.model->parameters()) {
	  grads.push_back(p.grad());
	}
	q_model.comm->allreduce(grads, true);
      }
      
      // optimizer step
      q_model.optimizer->step();
      q_model.lr_scheduler->step();
    }
  }

  // save loss values
  torch::Tensor q_loss_mean_tensor = q_loss_tensor;
  if (q_models[0].comm) {
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> q_loss_mean = {q_loss_tensor};
    q_models[0].comm->allreduce(q_loss_mean, true);
    q_loss_mean_tensor = q_loss_mean[0];
  }
  q_loss_val = q_loss_mean_tensor.item<T>();

  // policy function
  // freeze the q_models
  for (const auto& q_model : q_models) {
    set_grad_state(q_model.model, false);
  }

  // set p_model to train
  p_model.model->train();

  torch::Tensor action_old_pred_tensor, action_old_pred_log_prob;
  std::tie(action_old_pred_tensor, action_old_pred_log_prob) = (p_model.model)->forwardNoise(state_old_tensor);

  // just q1 is used
  // attention: we need to use gradient ASCENT on L here, which means we need to do gradient DESCENT on -L
  torch::Tensor q_tens =
    torch::squeeze(q_models[0].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_pred_tensor})[0], 1);
  for (int i = 1; i < q_models_target.size(); ++i) {
    auto q_tmp_tensor =
      torch::squeeze(q_models_target[i].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_tensor})[0], 1);
    q_tens = torch::minimum(q_tens, q_tmp_tensor);
  }
  // entropy regularization
  q_tens = q_tens - alpha_model->forward(action_old_pred_log_prob);

  // compute loss
  torch::Tensor p_loss_tensor = -torch::mean(q_tens);

  // bwd pass
  state = p_model.state;
  if ((state->step_train_current + 1) % p_model.grad_accumulation_steps == 0) {
    p_model.optimizer->zero_grad();
  }
  p_loss_tensor.backward();

  // allreduce (average) gradients (if running distributed)
  if ((state->step_train_current + 1) % p_model.grad_accumulation_steps == 0) {
    if (p_model.comm) {
      std::vector<torch::Tensor> grads;
      grads.reserve(p_model.model->parameters().size());
      for (const auto& p : p_model.model->parameters()) {
	grads.push_back(p.grad());
      }
      p_model.comm->allreduce(grads, true);
    }

    // optimizer step
    p_model.optimizer->step();
    p_model.lr_scheduler->step();
  }

  // unfreeze the qmodels
  for (const auto& q_model : q_models) {
    set_grad_state(q_model.model, true);
  }

  // save loss val
  p_loss_val = p_loss_tensor.item<T>();

  // do polyak averaging: only if we also trained the policy
  for (int i = 0; i < q_models_target.size(); ++i) {
    polyak_update<T>(q_models_target[i].model, q_models[i].model, rho);
  }

  // print some info
  // value functions
  for (const auto& q_model : q_models) {
    state = q_model.state;
    state->step_train++;
    state->step_train_current++;
  }
  auto q_model = q_models[0];
  state = q_models[0].state;
  if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
    std::stringstream os;
    os << "model: critic,";
    os << "step_train: " << state->step_train << ", ";
    os << "loss: " << q_loss_val << ", ";
    auto lrs = get_current_lrs(q_model.optimizer);
    os << "lr: " << lrs[0];
    if (!q_model.comm || (q_model.comm && q_model.comm->rank == 0)) {
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (state->enable_wandb_hook) {
        torchfort::wandb_log(state, q_model.comm, "critic_0", "train_loss", state->step_train, q_loss_val);
        torchfort::wandb_log(state, q_model.comm, "critic_0", "train_lr", state->step_train, lrs[0]);
      }
    }
  }

  // policy function
  state = p_model.state;
  state->step_train++;
  state->step_train_current++;
  if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
    std::stringstream os;
    os << "model: actor, ";
    os << "step_train: " << state->step_train << ", ";
    os << "loss: " << p_loss_val << ", ";
    auto lrs = get_current_lrs(p_model.optimizer);
    os << "lr: " << lrs[0];
    if (!p_model.comm || (p_model.comm && p_model.comm->rank == 0)) {
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (state->enable_wandb_hook) {
        torchfort::wandb_log(state, p_model.comm, "actor", "train_loss", state->step_train, p_loss_val);
        torchfort::wandb_log(state, p_model.comm, "actor", "train_lr", state->step_train, lrs[0]);
      }
    }

    if (alpha_optimizer) {
      float alpha_loss_val = alpha_loss.item<T>();
      std::stringstream osa;
      osa << "model: alpha, ";
      osa << "step_train: " << state->step_train << ", ";
      osa << "loss: " << alpha_loss_val << ", ";
      auto lrs = get_current_lrs(alpha_optimizer);
      osa << "lr: " << lrs[0];
      if (!p_model.comm || (p_model.comm && p_model.comm->rank == 0)) {
        torchfort::logging::print(osa.str(), torchfort::logging::info);
        if (state->enable_wandb_hook) {
          torchfort::wandb_log(state, p_model.comm, "alpha", "train_loss", state->step_train, alpha_loss_val);
          torchfort::wandb_log(state, p_model.comm, "alpha", "train_lr", state->step_train, lrs[0]);
        }
      }
    }
  }

  torchfort::nvtx::rangePop();
}

// SAC training system
class SACSystem : public RLOffPolicySystem, public std::enable_shared_from_this<RLOffPolicySystem> {

public:
  // constructor
  SACSystem(const char* name, const YAML::Node& system_node, int model_device, int rb_device);

  // init communicators
  void initSystemComm(MPI_Comm mpi_comm);

  // we should pass a tuple (s, a, s', r, d)
  // single env
  void updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, float r, bool d);
  // multi env
  void updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, torch::Tensor r, torch::Tensor d);
  void setSeed(unsigned int seed);
  bool isReady();

  // train step
  void trainStep(float& p_loss_val, float& q_loss_val);

  // predictions
  torch::Tensor explore(torch::Tensor action);
  torch::Tensor predict(torch::Tensor state);
  torch::Tensor predictExplore(torch::Tensor state);
  torch::Tensor evaluate(torch::Tensor state, torch::Tensor action);

  // saving and loading
  void saveCheckpoint(const std::string& checkpoint_dir) const;
  void loadCheckpoint(const std::string& checkpoint_dir);

  // info printing
  void printInfo() const;

  // accessors
  torch::Device modelDevice() const;
  torch::Device rbDevice() const;
  int getRank() const;

private:
  // we need those accessors for logging
  std::shared_ptr<ModelState> getSystemState_();

  std::shared_ptr<Comm> getSystemComm_();

  // models
  PolicyPack p_model_;
  std::vector<ModelPack> q_models_, q_models_target_;

  // replay buffer
  std::shared_ptr<ReplayBuffer> replay_buffer_;

  // system state
  std::shared_ptr<ModelState> system_state_;

  // system comm
  std::shared_ptr<Comm> system_comm_;

  // some parameters
  int batch_size_;
  int num_critics_;
  int nstep_;
  RewardReductionMode nstep_reward_reduction_;
  float gamma_;
  float rho_;
  float a_low_, a_high_;
  std::shared_ptr<AlphaModel> alpha_model_;
  std::shared_ptr<torch::optim::Optimizer> alpha_optimizer_;
  std::shared_ptr<BaseLRScheduler> alpha_lr_scheduler_;
  float target_entropy_;
};

} // namespace off_policy

} // namespace rl

} // namespace torchfort
