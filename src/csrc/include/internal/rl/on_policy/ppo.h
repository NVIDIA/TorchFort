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
#include "internal/rl/on_policy.h"
#include "internal/rl/policy.h"
#include "internal/rl/rollout_buffer.h"
#include "internal/rl/utils.h"

namespace torchfort {

namespace rl {

namespace on_policy {

// implementing https://spinningup.openai.com/en/latest/algorithms/ppo.html?highlight=PPO#id8
template <typename T>
void train_ppo(const ACPolicyPack& pq_model, torch::Tensor state_tensor, torch::Tensor action_tensor,
               torch::Tensor q_tensor, torch::Tensor log_p_tensor, torch::Tensor adv_tensor, torch::Tensor ret_tensor,
               const T& epsilon, const T& clip_q, const T& entropy_loss_coeff, const T& q_loss_coeff,
               const T& max_grad_norm, const T& target_kl_divergence, bool normalize_advantage, T& p_loss_val,
               T& q_loss_val, T& kl_divergence, T& clip_fraction, T& explained_var) {

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

  // normalize advantages if requested
  if (normalize_advantage && (batch_size > 1)) {
    // make sure we are not going to compute gradients
    torch::NoGradGuard no_grad;

    // compute mean
    torch::Tensor adv_mean = torch::sum(adv_tensor);
    auto options = torch::TensorOptions().dtype(torch::kLong).device(adv_mean.device());
    torch::Tensor adv_count = torch::tensor({torch::numel(adv_tensor)}, options);

    // average mean across all nodes
    if (pq_model.comm) {
      std::vector<torch::Tensor> means = {adv_mean, adv_count};
      pq_model.comm->allreduce(means, false);
      adv_mean = means[0];
      adv_count = means[1];
    }
    adv_mean = adv_mean / adv_count;

    // compute std
    torch::Tensor adv_std = torch::sum(torch::square(adv_tensor - adv_mean));

    // average std across all nodes
    if (pq_model.comm) {
      std::vector<torch::Tensor> stds = {adv_std};
      pq_model.comm->allreduce(stds, false);
      adv_std = stds[0];
    }
    adv_std = torch::sqrt(adv_std / (adv_count - 1));

    // update advantage tensor
    adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1.e-8);
  }

  // set models to train
  pq_model.model->train();

  // evaluate policies
  torch::Tensor log_p_new_tensor, entropy_tensor, q_new_tensor;
  std::tie(log_p_new_tensor, entropy_tensor, q_new_tensor) =
      pq_model.model->evaluateAction(state_tensor, action_tensor);

  // compute policy ratio
  torch::Tensor log_ratio_tensor = log_p_new_tensor - log_p_tensor;
  torch::Tensor ratio_tensor = torch::exp(log_ratio_tensor);

  // clipped surrogate loss
  torch::Tensor p_loss_tensor_1 = adv_tensor * ratio_tensor;
  torch::Tensor p_loss_tensor_2 = adv_tensor * torch::clamp(ratio_tensor, 1. - epsilon, 1. + epsilon);
  // the stable baselines code uses torch.min but I think this is wrong, it has to be torch.minimum
  torch::Tensor p_loss_tensor = -torch::mean(torch::minimum(p_loss_tensor_1, p_loss_tensor_2));

  // clip value function if requested
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

  // compute kl_divergence before doing anything which could destroy the model
  {
    torch::NoGradGuard no_grad;

    // kl_divergence
    torch::Tensor kl_divergence_tensor = torch::mean((ratio_tensor - 1.) - log_ratio_tensor);
    if (pq_model.comm) {
      std::vector<torch::Tensor> kl_divergence_mean = {kl_divergence_tensor};
      pq_model.comm->allreduce(kl_divergence_mean, true);
      kl_divergence_tensor = kl_divergence_mean[0];
    }
    kl_divergence = kl_divergence_tensor.item<T>();
  }

  // do saftey check to catch breaking training
  bool skip_step = false;
  if ((target_kl_divergence > 0.) && (kl_divergence > 1.5 * target_kl_divergence)) {
    skip_step = true;
  }

  // backward pass: only perform if we do not skip the step
  if (!skip_step) {
    auto state = pq_model.state;
    if (state->step_train_current % pq_model.grad_accumulation_steps == 0) {
      pq_model.optimizer->zero_grad();
    }
    // we need to be careful here: if we decide to skip the step, we might not want
    // to gradient accumulate here, so better skip backward pass entirely
    loss_tensor.backward();

    // weight update: only trigger if this is the last step
    // of the accumulation procedure
    if ((state->step_train_current + 1) % pq_model.grad_accumulation_steps == 0) {
      if (pq_model.comm) {
	std::vector<torch::Tensor> grads;
	grads.reserve(pq_model.model->parameters().size());
	for (const auto& p : pq_model.model->parameters()) {
	  grads.push_back(p.grad());
	}
	pq_model.comm->allreduce(grads, true);
      }
      
      // clip
      if (max_grad_norm > 0.) {
	torch::nn::utils::clip_grad_norm_(pq_model.model->parameters(), max_grad_norm);
      }

      // optimizer step
      // policy
      pq_model.optimizer->step();
      pq_model.lr_scheduler->step();
    }
  }

  // reduce losses across ranks for printing
  torch::Tensor p_loss_mean_tensor = p_loss_tensor;
  if (pq_model.comm) {
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> p_loss_mean = {p_loss_tensor};
    pq_model.comm->allreduce(p_loss_mean, true);
    p_loss_mean_tensor = p_loss_mean[0];
  }
  p_loss_val = p_loss_mean_tensor.item<T>();

  torch::Tensor q_loss_mean_tensor = q_loss_tensor;
  if (pq_model.comm) {
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> q_loss_mean = {q_loss_tensor};
    pq_model.comm->allreduce(q_loss_mean, true);
    q_loss_mean_tensor = q_loss_mean[0];
  }
  q_loss_val = q_loss_mean_tensor.item<T>();

  // policy function
  auto state = pq_model.state;
  if (!skip_step) {
    state->step_train++;
    state->step_train_current++;
  }
  if ((state->report_frequency > 0) && (state->step_train % state->report_frequency == 0)) {
    std::stringstream os;
    os << "model: "
       << "actor_critic"
       << ", ";
    os << "step_train: " << state->step_train << ", ";
    os << "p_loss: " << p_loss_val << ", ";
    os << "q_loss: " << q_loss_val << ", ";
    auto lrs = get_current_lrs(pq_model.optimizer);
    os << "lr: " << lrs[0] << ", ";
    os << "skipped: " << skip_step;
    if ((!pq_model.comm || (pq_model.comm && pq_model.comm->rank == 0)) && !skip_step) {
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (state->enable_wandb_hook) {
        torchfort::wandb_log(pq_model.state, pq_model.comm, "actor_critic", "train_loss_p", state->step_train,
                             p_loss_val);
        torchfort::wandb_log(pq_model.state, pq_model.comm, "actor_critic", "train_loss_q", state->step_train,
                             q_loss_val);
        torchfort::wandb_log(pq_model.state, pq_model.comm, "actor_critic", "train_lr", state->step_train, lrs[0]);
      }
    }
  }

  // some other diagnostic variables
  {
    torch::NoGradGuard no_grad;

    // clip_fraction
    torch::Tensor clip_fraction_tensor = torch::mean((torch::abs(ratio_tensor - 1.) > epsilon).to(torch::kFloat32));
    if (pq_model.comm) {
      std::vector<torch::Tensor> clip_fraction_mean = {clip_fraction_tensor};
      pq_model.comm->allreduce(clip_fraction_mean, true);
      clip_fraction_tensor = clip_fraction_mean[0];
    }
    clip_fraction = clip_fraction_tensor.item<T>();

    // explained variance
    torch::Tensor expvar_tensor = explained_variance(q_tensor, ret_tensor, pq_model.comm);
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

  // we should pass a tuple (s, a, r, e), single env convenience function
  void updateRolloutBuffer(torch::Tensor, torch::Tensor, float, bool);
  // this is for multi env
  void updateRolloutBuffer(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
  // void finalizeRolloutBuffer(float, bool);
  void resetRolloutBuffer();
  void setSeed(unsigned int);
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
  ACPolicyPack pq_model_;

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
  float max_grad_norm_;
  bool normalize_advantage_;
  ActorNormalizationMode actor_normalization_mode_;
};

} // namespace on_policy

} // namespace rl

} // namespace torchfort
