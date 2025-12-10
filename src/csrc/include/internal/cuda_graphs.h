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

#ifdef ENABLE_GPU
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <sstream>
#include <vector>

#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/exceptions.h"

namespace torchfort {

// Action to take for current iteration
enum class GraphAction {
  WARMUP,  // Run eager execution, increment warmup count
  CAPTURE, // Run eager execution with graph capture
  REPLAY   // Skip eager execution, replay captured graph
};

#ifdef ENABLE_GPU

// Number of warmup iterations before CUDA graph capture
constexpr int kCudaGraphWarmupIters = 3;

// RAII wrapper for cudaGraph_t
class CudaGraph {
public:
  CudaGraph() : graph_(nullptr) {}
  ~CudaGraph() {
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  // Non-copyable
  CudaGraph(const CudaGraph&) = delete;
  CudaGraph& operator=(const CudaGraph&) = delete;

  // Movable
  CudaGraph(CudaGraph&& other) noexcept : graph_(other.graph_) { other.graph_ = nullptr; }
  CudaGraph& operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
      if (graph_)
        cudaGraphDestroy(graph_);
      graph_ = other.graph_;
      other.graph_ = nullptr;
    }
    return *this;
  }

  cudaGraph_t& get() { return graph_; }
  cudaGraph_t get() const { return graph_; }
  bool valid() const { return graph_ != nullptr; }

private:
  cudaGraph_t graph_;
};

// RAII wrapper for cudaGraphExec_t
class CudaGraphExec {
public:
  CudaGraphExec() : exec_(nullptr) {}
  ~CudaGraphExec() {
    if (exec_) {
      cudaGraphExecDestroy(exec_);
    }
  }

  // Non-copyable
  CudaGraphExec(const CudaGraphExec&) = delete;
  CudaGraphExec& operator=(const CudaGraphExec&) = delete;

  // Movable
  CudaGraphExec(CudaGraphExec&& other) noexcept : exec_(other.exec_) { other.exec_ = nullptr; }
  CudaGraphExec& operator=(CudaGraphExec&& other) noexcept {
    if (this != &other) {
      if (exec_)
        cudaGraphExecDestroy(exec_);
      exec_ = other.exec_;
      other.exec_ = nullptr;
    }
    return *this;
  }

  cudaGraphExec_t& get() { return exec_; }
  cudaGraphExec_t get() const { return exec_; }
  bool valid() const { return exec_ != nullptr; }

private:
  cudaGraphExec_t exec_;
};

// Graph state for inference
class InferenceGraphState {
public:
  InferenceGraphState(const char* context = "inference") : context_(context) {}

  // Determine action for this iteration - validates inputs if captured, stores signature if ready to capture
  // Returns the action to take. Call begin_capture() after this if action == CAPTURE.
  GraphAction prepare(const std::vector<torch::Tensor>& inputs) {
    InputSignature current_sig = make_input_signature(inputs);

    if (captured_) {
      validate_inputs(current_sig);
      return GraphAction::REPLAY;
    }

    if (warmup_count_ == kCudaGraphWarmupIters) {
      input_signature_ = std::move(current_sig);
      return GraphAction::CAPTURE;
    }

    return GraphAction::WARMUP;
  }

  // Begin graph capture - call this after prepare() returns CAPTURE and after any pre-capture work
  void begin_capture(cudaStream_t capture_stream, cudaStream_t user_stream, c10::cuda::OptionalCUDAStreamGuard& guard,
                     int device_index) {
    CHECK_CUDA(cudaStreamSynchronize(user_stream));
    auto capture_c10_stream = c10::cuda::getStreamFromExternal(capture_stream, device_index);
    guard.reset_stream(capture_c10_stream);
    CHECK_CUDA(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
  }

  // Finalize after forward pass - handles capture end or warmup increment
  void finalize(GraphAction action, cudaStream_t capture_stream, cudaStream_t user_stream,
                c10::cuda::OptionalCUDAStreamGuard& guard, int device_index,
                const std::vector<torch::Tensor>& outputs) {
    if (action == GraphAction::CAPTURE) {
      end_capture(capture_stream, user_stream, guard, device_index);
      static_outputs_ = outputs;
    } else if (action == GraphAction::WARMUP) {
      warmup_count_++;
    }
  }

  // Launch captured graph on the given stream
  void launch(cudaStream_t stream) { CHECK_CUDA(cudaGraphLaunch(graph_exec_.get(), stream)); }

  // Get static outputs (valid after CAPTURE or REPLAY)
  const std::vector<torch::Tensor>& get_outputs() const { return static_outputs_; }

  bool is_captured() const { return captured_; }

private:
  // Input signature for validating consistent inputs
  struct InputSignature {
    std::vector<void*> ptrs;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<c10::ScalarType> dtypes;

    bool operator!=(const InputSignature& other) const {
      return ptrs != other.ptrs || shapes != other.shapes || dtypes != other.dtypes;
    }
  };

  static InputSignature make_input_signature(const std::vector<torch::Tensor>& tensors) {
    InputSignature sig;
    sig.ptrs.reserve(tensors.size());
    sig.shapes.reserve(tensors.size());
    sig.dtypes.reserve(tensors.size());
    for (const auto& t : tensors) {
      sig.ptrs.push_back(t.data_ptr());
      sig.shapes.push_back(t.sizes().vec());
      sig.dtypes.push_back(t.scalar_type());
    }
    return sig;
  }

  void validate_inputs(const InputSignature& current_sig) const {
    if (input_signature_ != current_sig) {
      std::stringstream ss;
      ss << "CUDA graph input mismatch in " << context_ << ". "
         << "When enable_cuda_graphs is set, input tensors must have consistent "
         << "data pointers, shapes, and dtypes across all calls. "
         << "If you need to change inputs, disable enable_cuda_graphs.";
      THROW_INVALID_USAGE(ss.str());
    }
  }

  void end_capture(cudaStream_t capture_stream, cudaStream_t user_stream, c10::cuda::OptionalCUDAStreamGuard& guard,
                   int device_index) {
    CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph_.get()));
    instantiate_graph();
    captured_ = true;
    auto user_c10_stream = c10::cuda::getStreamFromExternal(user_stream, device_index);
    guard.reset_stream(user_c10_stream);
  }

  void instantiate_graph() {
    cudaGraphNode_t error_node;
    char log_buffer[1024];
    cudaError_t result =
        cudaGraphInstantiate(&graph_exec_.get(), graph_.get(), &error_node, log_buffer, sizeof(log_buffer));
    if (result != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA graph instantiation failed in " << context_ << ": " << cudaGetErrorString(result);
      if (std::strlen(log_buffer) > 0) {
        ss << " Log: " << log_buffer;
      }
      THROW_INTERNAL_ERROR(ss.str());
    }
  }

  const char* context_;
  int warmup_count_ = 0;
  bool captured_ = false;
  InputSignature input_signature_;
  CudaGraph graph_;
  CudaGraphExec graph_exec_;
  std::vector<torch::Tensor> static_outputs_;
};

// Graph state for training (single graph for forward + loss + backward)
class TrainingGraphState {
public:
  TrainingGraphState(const char* context = "training") : context_(context) {}

  // Determine action for this iteration - validates inputs if captured, stores signature if ready to capture
  // Returns the action to take. Call begin_capture() after this if action == CAPTURE.
  GraphAction prepare(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& labels,
                      const std::vector<torch::Tensor>& extra_args) {
    InputSignature current_sig = make_input_signature(inputs, labels, extra_args);

    if (captured_) {
      validate_inputs(current_sig);
      return GraphAction::REPLAY;
    }

    if (warmup_count_ == kCudaGraphWarmupIters) {
      input_signature_ = std::move(current_sig);
      return GraphAction::CAPTURE;
    }

    return GraphAction::WARMUP;
  }

  // Begin graph capture - call this after prepare() returns CAPTURE and after any pre-capture work
  void begin_capture(cudaStream_t capture_stream, cudaStream_t user_stream, c10::cuda::OptionalCUDAStreamGuard& guard,
                     int device_index) {
    CHECK_CUDA(cudaStreamSynchronize(user_stream));
    auto capture_c10_stream = c10::cuda::getStreamFromExternal(capture_stream, device_index);
    guard.reset_stream(capture_c10_stream);
    CHECK_CUDA(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
  }

  // Finalize after forward+loss+backward pass - handles capture end or warmup increment
  void finalize(GraphAction action, cudaStream_t capture_stream, cudaStream_t user_stream,
                c10::cuda::OptionalCUDAStreamGuard& guard, int device_index, const torch::Tensor& loss) {
    if (action == GraphAction::CAPTURE) {
      end_capture(capture_stream, user_stream, guard, device_index);
      static_loss_ = loss;
    } else if (action == GraphAction::WARMUP) {
      warmup_count_++;
    }
  }

  // Launch captured graph on the given stream
  void launch(cudaStream_t stream) { CHECK_CUDA(cudaGraphLaunch(graph_exec_.get(), stream)); }

  // Get static loss (valid after CAPTURE or REPLAY)
  const torch::Tensor& get_loss() const { return static_loss_; }

  bool is_captured() const { return captured_; }

private:
  // Input signature for validating consistent inputs
  struct InputSignature {
    std::vector<void*> ptrs;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<c10::ScalarType> dtypes;

    bool operator!=(const InputSignature& other) const {
      return ptrs != other.ptrs || shapes != other.shapes || dtypes != other.dtypes;
    }
  };

  static InputSignature make_input_signature(const std::vector<torch::Tensor>& inputs,
                                             const std::vector<torch::Tensor>& labels,
                                             const std::vector<torch::Tensor>& extra_args) {
    InputSignature sig;
    size_t total = inputs.size() + labels.size() + extra_args.size();
    sig.ptrs.reserve(total);
    sig.shapes.reserve(total);
    sig.dtypes.reserve(total);

    for (const auto& t : inputs) {
      sig.ptrs.push_back(t.data_ptr());
      sig.shapes.push_back(t.sizes().vec());
      sig.dtypes.push_back(t.scalar_type());
    }
    for (const auto& t : labels) {
      sig.ptrs.push_back(t.data_ptr());
      sig.shapes.push_back(t.sizes().vec());
      sig.dtypes.push_back(t.scalar_type());
    }
    for (const auto& t : extra_args) {
      sig.ptrs.push_back(t.data_ptr());
      sig.shapes.push_back(t.sizes().vec());
      sig.dtypes.push_back(t.scalar_type());
    }
    return sig;
  }

  void validate_inputs(const InputSignature& current_sig) const {
    if (input_signature_ != current_sig) {
      std::stringstream ss;
      ss << "CUDA graph input mismatch in " << context_ << ". "
         << "When enable_cuda_graphs is set, input tensors must have consistent "
         << "data pointers, shapes, and dtypes across all calls. "
         << "If you need to change inputs, disable enable_cuda_graphs.";
      THROW_INVALID_USAGE(ss.str());
    }
  }

  void end_capture(cudaStream_t capture_stream, cudaStream_t user_stream, c10::cuda::OptionalCUDAStreamGuard& guard,
                   int device_index) {
    CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph_.get()));
    instantiate_graph();
    captured_ = true;
    auto user_c10_stream = c10::cuda::getStreamFromExternal(user_stream, device_index);
    guard.reset_stream(user_c10_stream);
  }

  void instantiate_graph() {
    cudaGraphNode_t error_node;
    char log_buffer[1024];
    cudaError_t result =
        cudaGraphInstantiate(&graph_exec_.get(), graph_.get(), &error_node, log_buffer, sizeof(log_buffer));
    if (result != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA graph instantiation failed in " << context_ << ": " << cudaGetErrorString(result);
      if (std::strlen(log_buffer) > 0) {
        ss << " Log: " << log_buffer;
      }
      THROW_INTERNAL_ERROR(ss.str());
    }
  }

  const char* context_;
  int warmup_count_ = 0;
  bool captured_ = false;
  InputSignature input_signature_;
  CudaGraph graph_;
  CudaGraphExec graph_exec_;
  torch::Tensor static_loss_;
};

// Graph state for a model, including the capture stream
class ModelGraphState {
public:
  InferenceGraphState inference{"inference"};
  TrainingGraphState training{"training"};

  ModelGraphState(int device_index = 0) : capture_stream_(nullptr), device_index_(device_index) {
    // Create a non-blocking stream for graph capture
    CHECK_CUDA(cudaSetDevice(device_index_));
    CHECK_CUDA(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
  }

  ~ModelGraphState() {
    if (capture_stream_) {
      cudaStreamDestroy(capture_stream_);
    }
  }

  // Non-copyable
  ModelGraphState(const ModelGraphState&) = delete;
  ModelGraphState& operator=(const ModelGraphState&) = delete;

  cudaStream_t capture_stream() const { return capture_stream_; }
  int device_index() const { return device_index_; }

private:
  cudaStream_t capture_stream_;
  int device_index_;
};

#endif

} // namespace torchfort
