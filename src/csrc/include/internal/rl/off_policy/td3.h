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
#include "internal/rl/noise_actor.h"
#include "internal/rl/off_policy.h"
#include "internal/rl/replay_buffer.h"
#include "internal/rl/utils.h"

namespace torchfort {

namespace rl {

namespace off_policy {

// implementing https://spinningup.openai.com/en/latest/algorithms/td3.html#pseudocode
// we implement the update on a single batch with (s, a, r, s', d):
// gamma is a tensor here to support multi-step delayed learning. Here, gamma^n
// is used where n is the number of delayed steps. since the episode can end in between,
// not all samples have the same n in some cases. this should account for that
// this routine is action scale agnostic, we just have to make sure that the scales produced by
// the policies is the same as expected by the q-functions
template <typename T>
void train_td3(const ModelPack& p_model, const ModelPack& p_model_target, const std::vector<ModelPack>& q_models,
               const std::vector<ModelPack>& q_models_target, torch::Tensor state_old_tensor,
               torch::Tensor state_new_tensor, torch::Tensor action_old_tensor, torch::Tensor action_new_tensor,
               torch::Tensor reward_tensor, torch::Tensor d_tensor, const T& gamma, const T& rho, T& p_loss_val,
               T& q_loss_val, bool update_policy) {

  // nvtx marker
  torchfort::nvtx::rangePush("torchfort_train_td3");

  // sanity checks
  // batch size
  auto batch_size = state_old_tensor.size(0);
  assert(batch_size == state_new_tensor.size(0));
  assert(batch_size == action_old_tensor.size(0));
  assert(batch_size == action_new_tensor.size(0));
  assert(batch_size == reward_tensor.size(0));
  assert(batch_size == d_tensor.size(0));
  // singleton dims
  assert(reward_tensor.size(1) == 1);
  assert(d_tensor.size(1) == 1);

  // value functions
  // set to train
  for (const auto& q_model : q_models) {
    q_model.model->train();
  }

  // opt
  // loss is fixed by algorithm
  auto q_loss_func = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));

  // policy function
  // compute y: use the target models for q_new, no grads
  torch::Tensor y_tensor;
  {
    torch::NoGradGuard no_grad;
    torch::Tensor q_new_tensor =
      torch::squeeze(q_models_target[0].model->forward(std::vector<torch::Tensor>{state_new_tensor, action_new_tensor})[0], 1);
    for (int i = 1; i < q_models_target.size(); ++i) {
      torch::Tensor q_tmp_tensor =
	torch::squeeze(q_models_target[i].model->forward(std::vector<torch::Tensor>{state_new_tensor, action_new_tensor})[0], 1);
      q_new_tensor = torch::minimum(q_new_tensor, q_tmp_tensor);
    }
    y_tensor = torch::Tensor(reward_tensor + q_new_tensor * gamma * (1. - d_tensor));
  }
  
  // backward and update step
  // compute loss for critics and zero grads while we are at it
  torch::Tensor q_old_tensor =
    torch::squeeze(q_models[0].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_tensor})[0], 1);
  torch::Tensor q_loss_tensor = q_loss_func->forward(q_old_tensor, y_tensor);
  auto state = q_models[0].state;
  if (state->step_train_current % q_models[0].grad_accumulation_steps == 0) {
    q_models[0].optimizer->zero_grad();
  }
  for (int i = 1; i < q_models.size(); ++i) {
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
  if (update_policy) {
    // freeze the q1model
    set_grad_state(q_models[0].model, false);

    // set p_model to train
    p_model.model->train();
    torch::Tensor action_old_pred_tensor = p_model.model->forward(std::vector<torch::Tensor>{state_old_tensor})[0];
    // just q1 is used
    // attention: we need to use gradient ASCENT on L here, which means we need to do gradient DESCENT on -L
    torch::Tensor p_loss_tensor = -torch::mean(
        q_models[0].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_pred_tensor})[0]);

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
    
    // unfreeze the q1model
    set_grad_state(q_models[0].model, true);

    // reduce losses across ranks for printing
    torch::Tensor p_loss_mean_tensor = p_loss_tensor;
    if (p_model.comm) {
      torch::NoGradGuard no_grad;
      std::vector<torch::Tensor> p_loss_mean = {p_loss_tensor};
      p_model.comm->allreduce(p_loss_mean, true);
      p_loss_mean_tensor = p_loss_mean[0];
    }
    p_loss_val = p_loss_mean_tensor.item<T>();
  } else {
    // make sure that the loss value is sane and not some garbage number
    p_loss_val = static_cast<T>(0.);
  }

  // do polyak averaging: only if we also trained the policy
  state = p_model.state;
  if ( update_policy && ((state->step_train_current + 1) % p_model.grad_accumulation_steps == 0) ) {
    for (int i = 0; i < q_models_target.size(); ++i) {
      polyak_update<T>(q_models_target[i].model, q_models[i].model, rho);
    }
    polyak_update<T>(p_model_target.model, p_model.model, rho);
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
    os << "model: critic, ";
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
  if (update_policy) {
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
          torchfort::wandb_log(p_model.state, p_model.comm, "actor", "train_loss", state->step_train, p_loss_val);
          torchfort::wandb_log(p_model.state, p_model.comm, "actor", "train_lr", state->step_train, lrs[0]);
        }
      }
    }
  }

  torchfort::nvtx::rangePop();
}

// TD3 training system
class TD3System : public RLOffPolicySystem, public std::enable_shared_from_this<RLOffPolicySystem> {

public:
  // constructor
  TD3System(const char* name, const YAML::Node& system_node, int model_device, int rb_device);

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

  // internally used functions
  torch::Tensor predictWithNoiseTrain_(torch::Tensor state);

  // models
  ModelPack p_model_, p_model_target_;
  std::vector<ModelPack> q_models_, q_models_target_;

  // replay buffer
  std::shared_ptr<ReplayBuffer> replay_buffer_;

  // system state
  std::shared_ptr<ModelState> system_state_;

  // system comm
  std::shared_ptr<Comm> system_comm_;

  // noise actors
  std::shared_ptr<NoiseActor> noise_actor_train_;
  std::shared_ptr<NoiseActor> noise_actor_exploration_;

  // some parameters
  int batch_size_;
  int num_critics_;
  int policy_lag_;
  int nstep_;
  RewardReductionMode nstep_reward_reduction_;
  float gamma_;
  float rho_;
  float a_low_, a_high_;
};

} // namespace off_policy

} // namespace rl

} // namespace torchfort
