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
#include <vector>

#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <internal/defines.h>
#include <internal/logging.h>
#include <internal/nvtx.h>
#include <internal/utils.h>

// Forward declaration
std::vector<double> torchfort_model_get_current_lrs(const char* name);

namespace torchfort {

// Declaration of external global variables
extern std::unordered_map<std::string, ModelPack> models;

template <MemoryLayout L, typename T>
void inference(const char* name, T* input, size_t input_dim, int64_t* input_shape, T* output, size_t output_dim,
               int64_t* output_shape, cudaStream_t ext_stream = 0) {
  torchfort::nvtx::rangePush("torchfort_inference");

  torch::NoGradGuard no_grad;

  int device_id = get_device_id(input);
  if (get_device_id(output) != device_id) {
    THROW_INVALID_USAGE("input and output arrays must reside on the same device.");
  }

  c10::cuda::OptionalCUDAStreamGuard guard;
  if (device_id >= 0) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, device_id);
    guard.reset_stream(stream);
  }

  auto input_tensor = get_tensor<L>(input, input_dim, input_shape, device_id);
  auto output_tensor = get_tensor<L>(output, output_dim, output_shape, device_id);

  auto model = models[name].model.get();
  model->to(input_tensor.device());
  model->eval();
  auto results = model->forward(std::vector<torch::Tensor>{input_tensor});

  output_tensor.copy_(results[0].reshape(output_tensor.sizes()));
  models[name].state->step_inference++;
  torchfort::nvtx::rangePop();
}

template <MemoryLayout L, typename T>
void train(const char* name, T* input, size_t input_dim, int64_t* input_shape, T* label, size_t label_dim,
           int64_t* label_shape, T* loss_val, cudaStream_t ext_stream = 0) {
  torchfort::nvtx::rangePush("torchfort_train");

  if (!models[name].optimizer) {
    THROW_INVALID_USAGE("Training requires an optimizer, but optimizer block was missing in configuration file.");
  }

  if (!models[name].loss) {
    THROW_INVALID_USAGE("Training requires a loss function, but loss block was missing in configuration file.");
  }

  int device_id = get_device_id(input);
  if (get_device_id(label) != device_id) {
    THROW_INVALID_USAGE("input and label arrays must reside on the same device.");
  }

  c10::cuda::OptionalCUDAStreamGuard guard;
  if (device_id >= 0) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, device_id);
    guard.reset_stream(stream);
  }

  auto input_tensor = get_tensor<L>(input, input_dim, input_shape, device_id);
  auto label_tensor = get_tensor<L>(label, label_dim, label_shape, device_id);

  auto model = models[name].model.get();
  model->to(input_tensor.device());
  model->train();
  auto opt = models[name].optimizer.get();

  // fwd pass
  auto results = model->forward(std::vector<torch::Tensor>{input_tensor});
  auto losses =
      models[name].loss->forward(std::vector<torch::Tensor>{results[0]}, std::vector<torch::Tensor>{label_tensor});

  // extract loss
  *loss_val = losses[0].template item<T>();

  // bwd pass
  opt->zero_grad();
  for (const auto& l : losses) {
    l.backward();
  }

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

  opt->step();
  if (models[name].lr_scheduler) {
    models[name].lr_scheduler->step();
  }

  auto state = models[name].state.get();
  state->step_train++;
  if (state->report_frequency > 0 && state->step_train % state->report_frequency == 0) {
    std::stringstream os;
    os << "model: " << name << ", ";
    os << "step_train: " << state->step_train << ", ";
    os << "loss: " << *loss_val << ", ";
    auto lrs = torchfort_model_get_current_lrs(name);
    os << "lr: " << lrs[0];
    if (!models[name].comm || (models[name].comm && models[name].comm->rank == 0)) {
      torchfort::logging::print(os.str(), torchfort::logging::info);
      if (state->enable_wandb_hook)
        torchfort::wandb_log(name, "train_loss", state->step_train, *loss_val);
      torchfort::wandb_log(name, "train_lr", state->step_train, lrs[0]);
    }
  }

  torchfort::nvtx::rangePop();
}

} // namespace torchfort
