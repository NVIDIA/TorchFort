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
#include "internal/rl/rollout_buffer.h"
#include "internal/rl/on_policy.h"
#include "internal/rl/utils.h"
#include "internal/rl/policy.h"

namespace torchfort {

namespace rl {

namespace on_policy {

// implementing https://spinningup.openai.com/en/latest/algorithms/ppo.html?highlight=PPO#id8
template <typename T>
void train_ppo(const ACPolicyPack& p_model, const ModelPack& q_model,
               torch::Tensor state_tensor, torch::Tensor action_tensor, 
               torch::Tensor q_tensor, torch::Tensor log_p_tensor, torch::Tensor adv_tensor, torch::Tensor ret_tensor,
	       const T& epsilon, const T& clip_q, const T& entropy_loss_coeff, const T& q_loss_coeff, bool normalize_advantage,
	       T& p_loss_val, T& q_loss_val, T& kl_divergence, T& clip_fraction, T& explained_var) {

  // nvtx marker
  torchfort::nvtx::rangePush("torchfort_train_ppo");

  // sanity checks
  // batch size
  auto batch_size = state_tensor.size(0);
  assert(batch_size == action_tensor.size(0));
  assert(batch_size == q_tensor.size(0));
  assert(batch_size == log_p_tensor.size(0));
  assert(batch_size == adv_tensor.size(0));
  assert(batch_size == ret_tensor.size(0));
  // singleton dims
  assert(q_tensor.size(1) == 1);
  assert(log_p_tensor.size(1) == 1);
  assert(adv_tensor.size(1) == 1);
  assert(ret_tensor.size(1) == 1);

  //std::cout << "action_tensor " << action_tensor << std::endl;
  //std::cout << "state_tensor "	<< state_tensor << std::endl;
  //std::cout << "ret_tensor "	<< ret_tensor << std::endl;
  //std::cout << "adv_tensor "	<< adv_tensor << std::endl;
  //std::cout << "log_p_tensor "	<< log_p_tensor << std::endl;
  //std::cout << "q_tensor "	<< q_tensor << std::endl;

  // normalize advantages if requested
  if ( normalize_advantage && (batch_size > 1) ) {
    // make sure we are not going to compute gradients
    torch::NoGradGuard no_grad;
    
    // compute mean
    torch::Tensor adv_mean = torch::mean(adv_tensor);

    // average mean across all nodes
    if (p_model.comm) {
      std::vector<torch::Tensor> means = {adv_mean};
      p_model.comm->allreduce(means, true);
      adv_mean = means[0];
    }

    // compute std
    torch::Tensor adv_std = torch::mean(torch::square(adv_tensor - adv_mean));

    // average across all nodes
    if (p_model.comm) {
      std::vector<torch::Tensor> stds = {adv_std};
      p_model.comm->allreduce(stds, true);
      adv_std = stds[0];
    }
    adv_std = torch::sqrt(adv_std);

    // update advantage tensor
    adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1.e-8);
  }

  // set models to train
  q_model.model->train();
  p_model.model->train();
  
  // evaluate policies
  torch::Tensor log_p_new_tensor, entropy_tensor;
  std::tie(log_p_new_tensor, entropy_tensor) = p_model.model->evaluateAction(state_tensor, action_tensor);
  torch::Tensor q_new_tensor = q_model.model->forward(std::vector<torch::Tensor>{state_tensor, action_tensor})[0];

  // compute policy ratio
  torch::Tensor log_ratio_tensor = log_p_new_tensor - log_p_tensor;

  //std::cout << "log_p_new_tensor " << log_p_new_tensor << std::endl;
  //std::cout << "log_ratio_tensor " << log_ratio_tensor << std::endl;
  
  torch::Tensor ratio_tensor = torch::exp(log_ratio_tensor);

  // clipped surrogate loss
  torch::Tensor p_loss_tensor_1 = adv_tensor * ratio_tensor;
  torch::Tensor p_loss_tensor_2 = adv_tensor * torch::clamp(ratio_tensor, 1. - epsilon, 1. + epsilon);
  // the stable baselines code uses torch.min but I think this is wrong, it has to be torch.minimum
  torch::Tensor p_loss_tensor = -torch::mean(torch::minimum(p_loss_tensor_1, p_loss_tensor_2));

  //std::cout << "ratio_tensor "      << ratio_tensor << std::endl;
  //std::cout << "p_loss_tensor_1 "      << p_loss_tensor_1 << std::endl;
  //std::cout << "p_loss_tensor_2 "      << p_loss_tensor_2 << std::endl;

  //clip value function if requested
  torch::Tensor q_pred_tensor;
  if (clip_q > 0.) {
    q_pred_tensor = q_tensor + torch::clamp(q_new_tensor - q_tensor, -clip_q, clip_q);
  } else {
    q_pred_tensor = q_new_tensor;
  }
  
  // critic loss
  // loss function is fixed by algorithm
  auto q_loss_func = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
  torch::Tensor q_loss_tensor = q_loss_func(ret_tensor, q_pred_tensor);

  // entropy loss
  torch::Tensor entropy_loss_tensor = -torch::mean(entropy_tensor);

  // combined loss
  torch::Tensor loss_tensor = p_loss_tensor + q_loss_coeff * q_loss_tensor + entropy_loss_coeff * entropy_loss_tensor;

  // backward pass
  p_model.optimizer->zero_grad();
  q_model.optimizer->zero_grad();
  loss_tensor.backward();

  // allreduces
  // policy
  if (p_model.comm) {
    std::vector<torch::Tensor> grads;
    grads.reserve(p_model.model->parameters().size());
    for (const auto& p : p_model.model->parameters()) {
      grads.push_back(p.grad());
    }
    p_model.comm->allreduce(grads, true);
  }

  // critic
  if (q_model.comm) {
    std::vector<torch::Tensor> grads;
    grads.reserve(q_model.model->parameters().size());
    for (const auto& p : q_model.model->parameters()) {
      grads.push_back(p.grad());
    }
    q_model.comm->allreduce(grads, true);
  }

  // optimizer step
  // policy
  p_model.optimizer->step();
  p_model.lr_scheduler->step();
  // critic
  q_model.optimizer->step();
  q_model.lr_scheduler->step();

  // save loss vals
  p_loss_val = p_loss_tensor.item<T>();
  q_loss_val = q_loss_tensor.item<T>();
  
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
	torchfort::wandb_log(p_model.state, p_model.comm, "actor", "clip_fraction", state->step_train, clip_fraction);
	torchfort::wandb_log(p_model.state, p_model.comm, "actor", "kl_divergence", state->step_train, kl_divergence);
      }
    }
  }

  state = q_model.state;
  state->step_train++;
  if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
    std::stringstream os;
    os << "model: " << "critic" << ", ";
    os << "step_train: " << state->step_train << ", ";
    os << "loss: " << q_loss_val << ", ";
    auto lrs = get_current_lrs(q_model.optimizer);
    os << "lr: " << lrs[0];
    if (!q_model.comm || (q_model.comm && q_model.comm->rank == 0)) {
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (state->enable_wandb_hook) {
	torchfort::wandb_log(q_model.state, q_model.comm, "critic", "train_loss", state->step_train,
			     q_loss_val);
	torchfort::wandb_log(q_model.state, q_model.comm, "critic", "train_lr", state->step_train, lrs[0]);
      }
    }
  }

  // some diagnostic variables
  {
    torch::NoGradGuard no_grad;

    // kl_divergence
    kl_divergence = torch::mean((ratio_tensor - 1.) - log_ratio_tensor).item<T>();

    // clip_fraction
    torch::Tensor clip_fraction_tensor = torch::mean((torch::abs(ratio_tensor - 1.) > epsilon).to(torch::kFloat32));
    clip_fraction = clip_fraction_tensor.item<T>();

    // explained variance
    torch::Tensor expvar_tensor = explained_variance(q_tensor, ret_tensor);
    explained_var = expvar_tensor.item<T>();
  }

  torchfort::nvtx::rangePop();
}

// PPO training system
class PPOSystem : public RLOnPolicySystem, public std::enable_shared_from_this<RLOnPolicySystem> {

public:
  // constructor
  PPOSystem(const char* name, const YAML::Node& system_node, int model_device, int rb_device);

  // init communicators
  void initSystemComm(MPI_Comm mpi_comm);

  // we should pass a tuple (s, a, r, e)
  void updateRolloutBuffer(torch::Tensor, torch::Tensor, float, bool);
  //void finalizeRolloutBuffer(float, bool);
  void resetRolloutBuffer();
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

  // models
  ACPolicyPack p_model_;
  ModelPack q_model_;

  // replay buffer
  std::shared_ptr<RolloutBuffer> rollout_buffer_;

  // system state
  std::shared_ptr<ModelState> system_state_;

  // system comm
  std::shared_ptr<Comm> system_comm_;

  // some parameters
  int batch_size_;
  float epsilon_, clip_q_;
  float gamma_, gae_lambda_;
  float entropy_loss_coeff_, value_loss_coeff_;
  float target_kl_divergence_, current_kl_divergence_, explained_variance_;
  float clip_fraction_;
  float a_low_, a_high_;
  bool normalize_advantage_;
  ActorNormalizationMode actor_normalization_mode_;
};

} // namespace on_policy
  
} // namespace rl

} // namespace torchfort
