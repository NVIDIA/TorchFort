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
#include "internal/model_pack.h"
#include "internal/nvtx.h"
#include "internal/tensor_list.h"
#include "internal/utils.h"
#ifdef ENABLE_GPU
#include "internal/cuda_graphs.h"
#endif

namespace torchfort {
// Declaration of external global variables
extern std::unordered_map<std::string, ModelPack> models;

#ifdef ENABLE_GPU
// Number of warmup iterations before CUDA graph capture
constexpr int kCudaGraphWarmupIters = 3;

// Helper to instantiate a CUDA graph from a captured graph
void instantiate_graph(CudaGraph& graph, CudaGraphExec& exec) {
  cudaGraphNode_t error_node;
  char log_buffer[1024];
  cudaError_t result = cudaGraphInstantiate(&exec.get(), graph.get(), &error_node, log_buffer, sizeof(log_buffer));
  if (result != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA graph instantiation failed: " << cudaGetErrorString(result);
    if (strlen(log_buffer) > 0) {
      ss << " Log: " << log_buffer;
    }
    THROW_INTERNAL_ERROR(ss.str());
  }
}
#endif

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

#ifdef ENABLE_GPU
  c10::cuda::OptionalCUDAStreamGuard stream_guard;
  c10::cuda::OptionalCUDAGuard cuda_guard;
  set_device_and_stream(stream_guard, cuda_guard, model->device(), ext_stream);
#endif

  inputs->to(model->device());

  model->eval();

  std::vector<torch::Tensor> results;

#ifdef ENABLE_GPU
  // CUDA graph handling
  bool capturing = false;
  InferenceGraphState* graph_state = nullptr;
  cudaStream_t capture_stream = nullptr;

  if (models[name].state->enable_cuda_graphs && model->device().is_cuda() && models[name].graph_state) {

    graph_state = &models[name].graph_state->inference;
    capture_stream = models[name].graph_state->capture_stream();

    // Create input signature for validation
    InputSignature current_sig = make_input_signature(inputs->tensors);

    if (graph_state->captured) {

      validate_input_signature(graph_state->input_signature, current_sig, "inference");

    } else if (graph_state->warmup_count == kCudaGraphWarmupIters) {

      // Store input signature used during capture
      graph_state->input_signature = current_sig;

      // Synchronize user stream before capture
      CHECK_CUDA(cudaStreamSynchronize(user_stream));

      // Switch PyTorch to use our capture stream
      auto capture_c10_stream = models[name].graph_state->get_capture_cuda_stream();
      guard.reset_stream(capture_c10_stream);

      // Begin capture
      CHECK_CUDA(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
      capturing = true;
    }
  }
#endif

  // Forward pass
#ifdef ENABLE_GPU
  if (!graph_state || !graph_state->captured) {
#endif
    results = model->forward(inputs->tensors);
#ifdef ENABLE_GPU
    if (graph_state) graph_state->warmup_count++;
  }

  if (graph_state) {

    if (capturing) {
      // End capture and instantiate the graph
      CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph_state->graph.get()));
      instantiate_graph(graph_state->graph, graph_state->graph_exec);
      graph_state->static_outputs = results;
      graph_state->captured = true;

      // Switch back to user stream for replay and subsequent operations
      auto user_c10_stream = c10::cuda::getStreamFromExternal(user_stream, model->device().index());
      guard.reset_stream(user_c10_stream);
    }

    // Replay graph
    if (graph_state->captured) {
      graph_state->graph_exec.launch(user_stream);
      results = graph_state->static_outputs;
    }
  }
#endif

  // Copy results to output tensors
  for (size_t i = 0; i < results.size(); ++i) {
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
  c10::cuda::OptionalCUDAStreamGuard stream_guard;
  c10::cuda::OptionalCUDAGuard cuda_guard;
  set_device_and_stream(stream_guard, cuda_guard, model->device(), ext_stream);
#endif

  inputs->to(model->device());
  labels->to(model->device());
  if (extra_loss_args)
    extra_loss_args->to(model->device());

  model->train();
  auto opt = models[name].optimizer;
  auto state = models[name].state;

  torch::Tensor loss;

#ifdef ENABLE_GPU
  // CUDA graph handling
  bool capturing = false;
  TrainingGraphState* graph_state = nullptr;
  cudaStream_t capture_stream = nullptr;

  if (state->enable_cuda_graphs && model->device().is_cuda() && models[name].graph_state &&
      models[name].grad_accumulation_steps == 1) {

    // Note: CUDA graph capture for training is disabled if gradient accumulation is active

    graph_state = &models[name].graph_state->training;
    capture_stream = models[name].graph_state->capture_stream();

    // Create input signature for validation
    std::vector<torch::Tensor> extra_args_vec = extra_loss_args ? extra_loss_args->tensors : std::vector<torch::Tensor>();
    InputSignature current_sig = make_input_signature(inputs->tensors, labels->tensors, extra_args_vec);

    if (graph_state->captured) {

      validate_input_signature(graph_state->input_signature, current_sig, "training");

    } else if (graph_state->warmup_count == kCudaGraphWarmupIters) {

      // Store input signature used during capture
      graph_state->input_signature = current_sig;
      capturing = true;
    }
  }
#endif

  if (state->step_train_current % models[name].grad_accumulation_steps == 0) {
#ifdef ENABLE_GPU
    // Only explicitly call zero_grad for non-replay steps
    if (!graph_state || !graph_state->captured) {
#endif
      opt->zero_grad(/*set_to_none=*/true);
#ifdef ENABLE_GPU
    }
#endif
  }

#ifdef ENABLE_GPU
  if (capturing) {
    // Synchronize user stream before capture
    CHECK_CUDA(cudaStreamSynchronize(user_stream));

    // Switch PyTorch to use our capture stream
    auto capture_c10_stream = models[name].graph_state->get_capture_cuda_stream();
    guard.reset_stream(capture_c10_stream);

    // Begin capture on our non-blocking stream
    CHECK_CUDA(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
  }
#endif

  // Forward + loss + backward
#ifdef ENABLE_GPU
  if (!graph_state || !graph_state->captured) {
#endif
    auto fwd_results = model->forward(inputs->tensors);
    loss = models[name].loss->forward(fwd_results, labels->tensors,
                                      (extra_loss_args) ? extra_loss_args->tensors : std::vector<torch::Tensor>());
    loss.backward();
#ifdef ENABLE_GPU
    if (graph_state) graph_state->warmup_count++;
  }

  if (graph_state) {
    if (capturing) {
      // End graph capture and instantiate
      CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph_state->graph.get()));
      instantiate_graph(graph_state->graph, graph_state->graph_exec);
      graph_state->static_loss = loss;
      graph_state->captured = true;

      // Switch back to user stream for replay and subsequent operations
      auto user_c10_stream = c10::cuda::getStreamFromExternal(user_stream, model->device().index());
      guard.reset_stream(user_c10_stream);
    }

    // Replay graph
    if (graph_state->captured) {
      graph_state->graph_exec.launch(user_stream);
      loss = graph_state->static_loss;
    }
  }
#endif

  // Extract loss value
  *loss_val = loss.item<float>();

  // Optimizer step and related operations
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
  if (extra_loss_args)
    extra_loss_args->reset();

  torchfort::nvtx::rangePop();
}
} // namespace torchfort
