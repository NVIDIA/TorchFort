/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unordered_map>
#include <vector>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/logging.h"
#include "internal/nvtx.h"
#include "internal/tensor_list.h"
#include "internal/utils.h"

namespace torchfort {
// Declaration of external global variables
extern std::unordered_map<std::string, ModelPack> models;

void inference_multiarg(const char* name, torchfort_tensor_list_t inputs_in, torchfort_tensor_list_t outputs_in,
                        cudaStream_t ext_stream = 0) {
  torchfort::nvtx::rangePush("torchfort_inference");

  if (models.count(name) == 0) {
    THROW_INVALID_USAGE("Invalid model name provided.");
  }

  if (!inputs_in || !outputs_in) {
    THROW_INVALID_USAGE("Input and output tensor lists are required.");
  }

  auto inputs = static_cast<TensorList*>(inputs_in);
  auto outputs = static_cast<TensorList*>(outputs_in);

  torch::NoGradGuard no_grad;

  auto model = models[name].model;

#if ENABLE_GPU
  c10::cuda::OptionalCUDAStreamGuard guard;
  if (model->device().is_cuda()) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, model->device().index());
    guard.reset_stream(stream);
  }
#endif

  inputs->to(model->device());

  model->eval();
  auto results = model->forward(inputs->tensors);

  for (int i = 0; i < results.size(); ++i) {
    outputs->tensors[i].copy_(results[i].reshape(outputs->tensors[i].sizes()));
  }
  models[name].state->step_inference++;

  inputs->reset();

  torchfort::nvtx::rangePop();
}

void train_multiarg(const char* name, torchfort_tensor_list_t inputs_in, torchfort_tensor_list_t labels_in,
                    float* loss_val, torchfort_tensor_list_t extra_loss_args_in, cudaStream_t ext_stream = 0) {
  torchfort::nvtx::rangePush("torchfort_train");

  if (!inputs_in || !labels_in) {
    THROW_INVALID_USAGE("Input and label tensor lists are required.");
  }
  auto inputs = static_cast<TensorList*>(inputs_in);
  auto labels = static_cast<TensorList*>(labels_in);
  auto extra_loss_args = static_cast<TensorList*>(extra_loss_args_in);

  if (models.count(name) == 0) {
    THROW_INVALID_USAGE("Invalid model name provided.");
  }

  if (!models[name].optimizer) {
    THROW_INVALID_USAGE("Training requires an optimizer, but optimizer block was missing in configuration file.");
  }

  if (!models[name].loss) {
    THROW_INVALID_USAGE("Training requires a loss function, but loss block was missing in configuration file.");
  }

  auto model = models[name].model;

#ifdef ENABLE_GPU
  c10::cuda::OptionalCUDAStreamGuard guard;
  if (model->device().is_cuda()) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, model->device().index());
    guard.reset_stream(stream);
  }
#endif

  inputs->to(model->device());
  labels->to(model->device());
  if (extra_loss_args)
    extra_loss_args->to(model->device());

  model->train();
  auto opt = models[name].optimizer;
  auto state = models[name].state;

  // fwd pass
  auto results = model->forward(inputs->tensors);
  auto loss = models[name].loss->forward(results, labels->tensors,
                                         (extra_loss_args) ? extra_loss_args->tensors : std::vector<torch::Tensor>());

  // extract loss
  *loss_val = loss.item<float>();

  // bwd pass
  if (state->step_train_current % models[name].grad_accumulation_steps == 0) {
    opt->zero_grad();
  }

  loss.backward();

  if ((state->step_train_current + 1) % models[name].grad_accumulation_steps == 0) {
    // allreduce (average) gradients (if running distributed)
    if (models[name].comm) {
      std::vector<torch::Tensor> grads;
      grads.reserve(model->parameters().size());
      for (const auto& p : model->parameters()) {
        grads.push_back(p.grad());
      }
      models[name].comm->allreduce(grads, true);

      // average returned loss value
      models[name].comm->allreduce(*loss_val, true);
    }

    if (models[name].max_grad_norm > 0.0) {
      torch::nn::utils::clip_grad_norm_(model->parameters(), models[name].max_grad_norm);
    }

    opt->step();
    if (models[name].lr_scheduler) {
      models[name].lr_scheduler->step();
    }
  }

  state->step_train++;
  state->step_train_current++;
  if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
    std::stringstream os;
    os << "model: " << name << ", ";
    os << "step_train: " << state->step_train << ", ";
    os << "loss: " << *loss_val << ", ";
    auto lrs = get_current_lrs(name);
    os << "lr: " << lrs[0];
    if (!models[name].comm || (models[name].comm && models[name].comm->rank == 0)) {
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (state->enable_wandb_hook)
        torchfort::wandb_log(name, "train_loss", state->step_train, *loss_val);
      torchfort::wandb_log(name, "train_lr", state->step_train, lrs[0]);
    }
  }

  inputs->reset();
  labels->reset();

  torchfort::nvtx::rangePop();
}
} // namespace torchfort
