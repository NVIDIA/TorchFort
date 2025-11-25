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

#pragma once
#include <memory>
#include <vector>

#include <torch/torch.h>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include "internal/base_loss.h"
#include "internal/base_lr_scheduler.h"
#include "internal/distributed.h"
#include "internal/model_state.h"
#include "internal/model_wrapper.h"

namespace torchfort {

#ifdef ENABLE_GPU
// Structure to manage CUDA graph state for training and inference
struct CudaGraphState {
  // Inference graph
  cudaGraph_t inference_graph = nullptr;
  cudaGraphExec_t inference_graph_exec = nullptr;

  // Training graph (captures forward + loss + backward)
  cudaGraph_t train_graph = nullptr;
  cudaGraphExec_t train_graph_exec = nullptr;

  // Dedicated capture stream (required for graph capture, cannot use default stream)
  cudaStream_t capture_stream = nullptr;

  // Events for synchronization between user stream and capture stream
  cudaEvent_t sync_event = nullptr;

  // Static staging tensors for inference (fixed memory addresses for graph capture)
  std::vector<torch::Tensor> inference_static_inputs;
  std::vector<torch::Tensor> inference_static_outputs;

  // Static staging tensors for training (fixed memory addresses for graph capture)
  std::vector<torch::Tensor> train_static_inputs;
  std::vector<torch::Tensor> train_static_labels;
  std::vector<torch::Tensor> train_static_extra_loss_args;
  torch::Tensor train_static_loss;

  // Configuration
  bool enable_cuda_graphs = false;
  static constexpr int warmup_iterations = 3;

  // State tracking
  bool inference_graph_initialized = false;
  bool train_graph_initialized = false;

  int inference_iteration_count = 0;
  int train_iteration_count = 0;

  // Shape tracking for invalidation
  std::vector<std::vector<int64_t>> cached_inference_input_shapes;
  std::vector<std::vector<int64_t>> cached_train_input_shapes;
  std::vector<std::vector<int64_t>> cached_train_label_shapes;
  std::vector<std::vector<int64_t>> cached_train_extra_loss_args_shapes;

  // Destructor to clean up graphs, stream, and events
  ~CudaGraphState() {
    if (inference_graph_exec) cudaGraphExecDestroy(inference_graph_exec);
    if (inference_graph) cudaGraphDestroy(inference_graph);
    if (train_graph_exec) cudaGraphExecDestroy(train_graph_exec);
    if (train_graph) cudaGraphDestroy(train_graph);
    if (sync_event) cudaEventDestroy(sync_event);
    if (capture_stream) cudaStreamDestroy(capture_stream);
  }

  // Reset all graphs
  void reset_all() {
    if (inference_graph_exec) {
      cudaGraphExecDestroy(inference_graph_exec);
      inference_graph_exec = nullptr;
    }
    if (inference_graph) {
      cudaGraphDestroy(inference_graph);
      inference_graph = nullptr;
    }
    if (train_graph_exec) {
      cudaGraphExecDestroy(train_graph_exec);
      train_graph_exec = nullptr;
    }
    if (train_graph) {
      cudaGraphDestroy(train_graph);
      train_graph = nullptr;
    }

    inference_graph_initialized = false;
    train_graph_initialized = false;
    inference_iteration_count = 0;
    train_iteration_count = 0;
    cached_inference_input_shapes.clear();
    cached_train_input_shapes.clear();
    cached_train_label_shapes.clear();
    cached_train_extra_loss_args_shapes.clear();
    inference_static_inputs.clear();
    inference_static_outputs.clear();
    train_static_inputs.clear();
    train_static_labels.clear();
    train_static_extra_loss_args.clear();
    train_static_loss = torch::Tensor();
  }
};
#endif

// Simple struct to group model, optimizer, lr scheduler, state, and comm objects
struct ModelPack {
  std::shared_ptr<ModelWrapper> model;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
  std::shared_ptr<BaseLRScheduler> lr_scheduler;
  std::shared_ptr<BaseLoss> loss;
  std::shared_ptr<Comm> comm;
  std::shared_ptr<ModelState> state;
  int grad_accumulation_steps = 1;
  float max_grad_norm = 0.0;
#ifdef ENABLE_GPU
  std::shared_ptr<CudaGraphState> cuda_graph_state;
#endif
};

void save_model_pack(const ModelPack& model_pack, const std::string& fname, bool save_optimizer = true);
void load_model_pack(ModelPack& model_pack, const std::string& fname, bool load_optimizer = true);

} // namespace torchfort
