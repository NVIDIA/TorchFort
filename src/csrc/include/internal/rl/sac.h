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

#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/logging.h"
#include "internal/lr_schedulers.h"
#include "internal/model_pack.h"
#include "internal/setup.h"

// rl stuff
#include "internal/rl/replay_buffer.h"
#include "internal/rl/rl.h"
#include "internal/rl/utils.h"
#include "internal/rl/policy.h"

namespace torchfort {

namespace rl {

struct SACPolicyPack {
  std::shared_ptr<ACPolicy> model;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
  std::shared_ptr<BaseLRScheduler> lr_scheduler;
  std::shared_ptr<BaseLoss> loss;
  std::shared_ptr<Comm> comm;
  std::shared_ptr<ModelState> state;
};

// implementing https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
template <typename T>
void train_sac(const SACPolicyPack& p_model, const std::vector<ModelPack>& q_models,
               const std::vector<ModelPack>& q_models_target, torch::Tensor state_old_tensor,
               torch::Tensor state_new_tensor, torch::Tensor action_old_tensor, torch::Tensor reward_tensor,
               torch::Tensor d_tensor, const T& alpha, const T& gamma, const T& rho, T& p_loss_val,
               std::vector<T>& q_loss_vals) {

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

  // policy function
  // compute y: use the target models for q_new, no grads
  torch::Tensor y_tensor;
  {
    torch::NoGradGuard no_grad;
    torch::Tensor action_new_tensor, action_new_log_prob;
    std::tie(action_new_tensor, action_new_log_prob) = p_model.model->forwardNoise(state_new_tensor);

    // compute expected reward
    auto q_new_tensor =
        q_models_target[0].model->forward(std::vector<torch::Tensor>{state_new_tensor, action_new_tensor})[0];
    for (int i = 1; i < q_models_target.size(); ++i) {
      auto q_tmp_tensor =
          q_models_target[i].model->forward(std::vector<torch::Tensor>{state_new_tensor, action_new_tensor})[0];
      q_new_tensor = torch::minimum(q_new_tensor, q_tmp_tensor);
    }
    y_tensor = torch::Tensor(reward_tensor + (q_new_tensor - alpha * action_new_log_prob) * gamma * (1. - d_tensor));
  }

  // backward and update step
  q_loss_vals.clear();
  for (const auto& q_model : q_models) {
    // compute loss
    auto q_old_tensor = q_model.model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_tensor})[0];
    auto q_loss_tensor = q_loss_func->forward(q_old_tensor, y_tensor);
    q_model.optimizer->zero_grad();
    q_loss_tensor.backward();

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

    // save loss values
    q_loss_vals.push_back(q_loss_tensor.item<T>());
  }

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
  auto q_tens = q_models[0].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_pred_tensor})[0];
  for (int i = 1; i < q_models_target.size(); ++i) {
    auto q_tmp_tensor =
        q_models_target[i].model->forward(std::vector<torch::Tensor>{state_old_tensor, action_old_tensor})[0];
    q_tens = torch::minimum(q_tens, q_tmp_tensor);
  }
  torch::Tensor p_loss_tensor = -torch::mean(q_tens - action_old_pred_log_prob * alpha);
  // attention: we need to use gradient ASCENT on L here, which means we need to do gradient DESCENT on -L
  // p_loss_tensor = -p_loss_tensor;

  // bwd pass
  p_model.optimizer->zero_grad();
  p_loss_tensor.backward();

  // allreduce (average) gradients (if running distributed)
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
  for (int i = 0; i < q_models.size(); ++i) {
    auto q_model = q_models[i];
    auto state = q_models[i].state;
    std::string qname = "critic_" + std::to_string(i);
    state->step_train++;
    if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
      std::stringstream os;
      os << "model: " << qname << ", ";
      os << "step_train: " << state->step_train << ", ";
      os << "loss: " << q_loss_vals[i] << ", ";
      auto lrs = get_current_lrs(q_model.optimizer);
      os << "lr: " << lrs[0];
      if (!q_model.comm || (q_model.comm && q_model.comm->rank == 0)) {
        torchfort::logging::print(os.str(), torchfort::logging::info);
        if (state->enable_wandb_hook) {
          torchfort::wandb_log(q_model.state, q_model.comm, qname.c_str(), "train_loss", state->step_train,
                               q_loss_vals[i]);
          torchfort::wandb_log(q_model.state, q_model.comm, qname.c_str(), "train_lr", state->step_train, lrs[0]);
        }
      }
    }
  }

  // policy function
  auto state = p_model.state;
  state->step_train++;
  if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
    std::stringstream os;
    os << "model: "
       << "actor"
       << ", ";
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
  void updateReplayBuffer(torch::Tensor s, torch::Tensor a, torch::Tensor sp, float r, bool d);
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

private:
  // we need those accessors for logging
  std::shared_ptr<ModelState> getSystemState_();

  std::shared_ptr<Comm> getSystemComm_();

  // device
  torch::Device model_device_;
  torch::Device rb_device_;

  // models
  SACPolicyPack p_model_;
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
  float alpha_;
  float a_low_, a_high_;
};

} // namespace rl

} // namespace torchfort
