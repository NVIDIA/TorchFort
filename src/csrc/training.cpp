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

#ifdef ENABLE_GPU
// Helper function to extract shapes from tensor list
static std::vector<std::vector<int64_t>> get_tensor_shapes(const std::vector<torch::Tensor>& tensors) {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    shapes.push_back(tensor.sizes().vec());
  }
  return shapes;
}

// Helper function to check if shapes match
static bool shapes_match(const std::vector<std::vector<int64_t>>& shapes1,
                         const std::vector<std::vector<int64_t>>& shapes2) {
  if (shapes1.size() != shapes2.size()) {
    return false;
  }
  for (size_t i = 0; i < shapes1.size(); ++i) {
    if (shapes1[i] != shapes2[i]) {
      return false;
    }
  }
  return true;
}
#endif
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

#ifdef ENABLE_GPU
  c10::cuda::OptionalCUDAStreamGuard guard;
  cudaStream_t stream = 0;
  if (model->device().is_cuda()) {
    auto torch_stream = c10::cuda::getStreamFromExternal(ext_stream, model->device().index());
    guard.reset_stream(torch_stream);
    stream = torch_stream.stream();
  }
#endif

  inputs->to(model->device());

  model->eval();

#ifdef ENABLE_GPU
  auto& graph_state = models[name].cuda_graph_state;
  bool use_cuda_graphs = graph_state && graph_state->enable_cuda_graphs && model->device().is_cuda();

  if (use_cuda_graphs) {
    // Check if input shapes have changed
    auto current_shapes = get_tensor_shapes(inputs->tensors);
    bool shapes_changed = !graph_state->cached_inference_input_shapes.empty() &&
                          !shapes_match(current_shapes, graph_state->cached_inference_input_shapes);

    if (shapes_changed) {
      // Reset graph if shapes changed
      if (graph_state->inference_graph_exec) {
        cudaGraphExecDestroy(graph_state->inference_graph_exec);
        graph_state->inference_graph_exec = nullptr;
      }
      if (graph_state->inference_graph) {
        cudaGraphDestroy(graph_state->inference_graph);
        graph_state->inference_graph = nullptr;
      }
      graph_state->inference_graph_initialized = false;
      graph_state->inference_iteration_count = 0;
      graph_state->cached_inference_input_shapes.clear();
    }

    // Warmup iterations before capturing graph
    if (!graph_state->inference_graph_initialized &&
        graph_state->inference_iteration_count < graph_state->warmup_iterations) {
      auto results = model->forward(inputs->tensors);
      for (int i = 0; i < results.size(); ++i) {
        outputs->tensors[i].copy_(results[i].reshape(outputs->tensors[i].sizes()));
      }
      graph_state->inference_iteration_count++;
    } else {
      if (!graph_state->inference_graph_initialized) {
        // Capture graph on dedicated capture stream
        // Create static staging tensors with fixed memory addresses
        graph_state->inference_static_inputs.clear();
        graph_state->inference_static_outputs.clear();
        for (const auto& input_tensor : inputs->tensors) {
          graph_state->inference_static_inputs.push_back(
              torch::empty_like(input_tensor, input_tensor.options()));
        }
        for (const auto& output_tensor : outputs->tensors) {
          graph_state->inference_static_outputs.push_back(
              torch::empty_like(output_tensor, output_tensor.options()));
        }

        // Record event on user stream
        cudaEventRecord(graph_state->sync_event, stream);

        // Wait for user stream to complete on capture stream
        cudaStreamWaitEvent(graph_state->capture_stream, graph_state->sync_event, 0);

        // Begin graph capture on capture stream
        cudaStreamBeginCapture(graph_state->capture_stream, cudaStreamCaptureModeGlobal);

        // Temporarily switch PyTorch to use capture stream
        c10::cuda::CUDAStreamGuard capture_guard(
            c10::cuda::getStreamFromExternal(graph_state->capture_stream, model->device().index()));

        // Forward pass using static tensors
        auto results = model->forward(graph_state->inference_static_inputs);
        for (int i = 0; i < results.size(); ++i) {
          graph_state->inference_static_outputs[i].copy_(results[i].reshape(graph_state->inference_static_outputs[i].sizes()));
        }

        // End capture
        cudaStreamEndCapture(graph_state->capture_stream, &graph_state->inference_graph);

        // Instantiate graph
        cudaGraphInstantiate(&graph_state->inference_graph_exec,
                            graph_state->inference_graph, nullptr, nullptr, 0);

        graph_state->inference_graph_initialized = true;
        graph_state->cached_inference_input_shapes = current_shapes;

        if (models[name].state->verbose) {
          torchfort::logging::print("CUDA graph captured for inference", torchfort::logging::info);
        }

      }

      // Copy current input data to static tensors
      for (size_t i = 0; i < inputs->tensors.size(); ++i) {
        graph_state->inference_static_inputs[i].copy_(inputs->tensors[i]);
      }

      // Replay the graph
      cudaGraphLaunch(graph_state->inference_graph_exec, stream);

      // Synchronize to ensure graph completes
      cudaStreamSynchronize(stream);

      // Copy results from static tensors to user outputs
      for (size_t i = 0; i < outputs->tensors.size(); ++i) {
        outputs->tensors[i].copy_(graph_state->inference_static_outputs[i]);
      }
    }
  } else
#endif
  {
    // Non-graph path
    auto results = model->forward(inputs->tensors);
    for (int i = 0; i < results.size(); ++i) {
      outputs->tensors[i].copy_(results[i].reshape(outputs->tensors[i].sizes()));
    }
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
  cudaStream_t stream = 0;
  if (model->device().is_cuda()) {
    auto torch_stream = c10::cuda::getStreamFromExternal(ext_stream, model->device().index());
    guard.reset_stream(torch_stream);
    stream = torch_stream.stream();
  }
#endif

  inputs->to(model->device());
  labels->to(model->device());
  if (extra_loss_args)
    extra_loss_args->to(model->device());

  model->train();
  auto opt = models[name].optimizer;
  auto state = models[name].state;

#ifdef ENABLE_GPU
  auto& graph_state = models[name].cuda_graph_state;
  bool use_cuda_graphs = graph_state && graph_state->enable_cuda_graphs && model->device().is_cuda();

  // Check if input shapes have changed
  if (use_cuda_graphs) {
    auto current_input_shapes = get_tensor_shapes(inputs->tensors);
    auto current_label_shapes = get_tensor_shapes(labels->tensors);

    bool shapes_changed = (!graph_state->cached_train_input_shapes.empty() &&
                          (!shapes_match(current_input_shapes, graph_state->cached_train_input_shapes) ||
                           !shapes_match(current_label_shapes, graph_state->cached_train_label_shapes)));

    if (shapes_changed) {
      // Reset training graph if shapes changed
      if (graph_state->train_graph_exec) {
        cudaGraphExecDestroy(graph_state->train_graph_exec);
        graph_state->train_graph_exec = nullptr;
      }
      if (graph_state->train_graph) {
        cudaGraphDestroy(graph_state->train_graph);
        graph_state->train_graph = nullptr;
      }
      graph_state->train_graph_initialized = false;
      graph_state->train_iteration_count = 0;
      graph_state->cached_train_input_shapes.clear();
      graph_state->cached_train_label_shapes.clear();
    }
  }
#endif

  // Variable to store loss tensor
  torch::Tensor loss;

  // Forward + loss + backward pass (captured in a single graph when enabled)
#ifdef ENABLE_GPU
  if (use_cuda_graphs && models[name].grad_accumulation_steps == 1) {
    // Warmup iterations
    if (!graph_state->train_graph_initialized &&
        graph_state->train_iteration_count < graph_state->warmup_iterations) {
      opt->zero_grad(/*set_to_none=*/true);

      auto results = model->forward(inputs->tensors);
      loss = models[name].loss->forward(results, labels->tensors,
                                        (extra_loss_args) ? extra_loss_args->tensors : std::vector<torch::Tensor>());
      loss.backward();
      graph_state->train_iteration_count++;
    } else {
      if (!graph_state->train_graph_initialized) {
        // Capture graph on capture stream (forward + loss + backward)

        opt->zero_grad(/*set_to_none=*/true);

        // Create static staging tensors with fixed memory addresses
        graph_state->train_static_inputs.clear();
        graph_state->train_static_labels.clear();
        graph_state->train_static_extra_loss_args.clear();

        for (const auto& input_tensor : inputs->tensors) {
          graph_state->train_static_inputs.push_back(
              torch::empty_like(input_tensor, input_tensor.options()));
        }
        for (const auto& label_tensor : labels->tensors) {
          graph_state->train_static_labels.push_back(
              torch::empty_like(label_tensor, label_tensor.options()));
        }
        if (extra_loss_args) {
          for (const auto& extra_tensor : extra_loss_args->tensors) {
            graph_state->train_static_extra_loss_args.push_back(
                torch::empty_like(extra_tensor, extra_tensor.options()));
          }
        }

        auto current_input_shapes = get_tensor_shapes(inputs->tensors);
        auto current_label_shapes = get_tensor_shapes(labels->tensors);
        auto current_extra_loss_args_shapes = get_tensor_shapes(extra_loss_args->tensors);
        graph_state->cached_train_input_shapes = current_input_shapes;
        graph_state->cached_train_label_shapes = current_label_shapes;
        graph_state->cached_train_extra_loss_args_shapes = current_extra_loss_args_shapes;

        // Record event on user stream
        cudaEventRecord(graph_state->sync_event, stream);

        // Wait for user stream to complete on capture stream
        cudaStreamWaitEvent(graph_state->capture_stream, graph_state->sync_event, 0);

        // Begin graph capture on capture stream
        cudaStreamBeginCapture(graph_state->capture_stream, cudaStreamCaptureModeGlobal);

        // Temporarily switch PyTorch to use capture stream
        c10::cuda::CUDAStreamGuard capture_guard(
            c10::cuda::getStreamFromExternal(graph_state->capture_stream, model->device().index()));

        // Forward + loss + backward
        auto results = model->forward(graph_state->train_static_inputs);
        graph_state->train_static_loss = models[name].loss->forward(
            results, graph_state->train_static_labels,
            (extra_loss_args) ? graph_state->train_static_extra_loss_args : std::vector<torch::Tensor>());
        graph_state->train_static_loss.backward();

        // End capture
        cudaStreamEndCapture(graph_state->capture_stream, &graph_state->train_graph);

        // Instantiate graph
        cudaGraphInstantiate(&graph_state->train_graph_exec,
                            graph_state->train_graph, nullptr, nullptr, 0);

        graph_state->train_graph_initialized = true;

        if (state->verbose) {
          torchfort::logging::print("CUDA graph captured for training (forward + loss + backward)", torchfort::logging::info);
        }

      }

      // Copy current input/label/extra data to static tensors
      for (size_t i = 0; i < inputs->tensors.size(); ++i) {
        graph_state->train_static_inputs[i].copy_(inputs->tensors[i]);
      }
      for (size_t i = 0; i < labels->tensors.size(); ++i) {
        graph_state->train_static_labels[i].copy_(labels->tensors[i]);
      }
      if (extra_loss_args) {
        for (size_t i = 0; i < extra_loss_args->tensors.size(); ++i) {
          graph_state->train_static_extra_loss_args[i].copy_(extra_loss_args->tensors[i]);
        }
      }

      // Launch graph
      cudaGraphLaunch(graph_state->train_graph_exec, stream);

      // Synchronize to ensure graph completes before reading loss
      cudaStreamSynchronize(stream);

      // Get loss from static tensor
      loss = graph_state->train_static_loss;
    }
  } else
#endif
  {
    // Handle zero_grad before forward pass (for gradient accumulation)
    if (state->step_train_current % models[name].grad_accumulation_steps == 0) {
      opt->zero_grad(/*set_to_none=*/true);
    }
    // Non-graph path
    auto results = model->forward(inputs->tensors);
    loss = models[name].loss->forward(results, labels->tensors,
                                      (extra_loss_args) ? extra_loss_args->tensors : std::vector<torch::Tensor>());
    loss.backward();
  }

  // Extract loss value
  *loss_val = loss.item<float>();

  // Optimizer step and LR scheduler update (not graphed)
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

    // Gradient clipping
    if (models[name].max_grad_norm > 0.0) {
      torch::nn::utils::clip_grad_norm_(model->parameters(), models[name].max_grad_norm);
    }

    // Optimizer step (always outside graph)
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
