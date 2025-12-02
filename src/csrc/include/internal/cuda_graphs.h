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

#include <cuda_runtime.h>

#include <sstream>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/exceptions.h"

namespace torchfort {

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
  CudaGraph(CudaGraph&& other) noexcept : graph_(other.graph_) {
    other.graph_ = nullptr;
  }
  CudaGraph& operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
      if (graph_) cudaGraphDestroy(graph_);
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
  CudaGraphExec(CudaGraphExec&& other) noexcept : exec_(other.exec_) {
    other.exec_ = nullptr;
  }
  CudaGraphExec& operator=(CudaGraphExec&& other) noexcept {
    if (this != &other) {
      if (exec_) cudaGraphExecDestroy(exec_);
      exec_ = other.exec_;
      other.exec_ = nullptr;
    }
    return *this;
  }

  cudaGraphExec_t& get() { return exec_; }
  cudaGraphExec_t get() const { return exec_; }
  bool valid() const { return exec_ != nullptr; }

  // Launch the graph on a stream
  void launch(cudaStream_t stream) {
    if (exec_) {
      CHECK_CUDA(cudaGraphLaunch(exec_, stream));
    }
  }

private:
  cudaGraphExec_t exec_;
};

// Input signature for validating consistent inputs
struct InputSignature {
  std::vector<void*> ptrs;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<c10::ScalarType> dtypes;

  bool operator==(const InputSignature& other) const {
    return ptrs == other.ptrs && shapes == other.shapes && dtypes == other.dtypes;
  }

  bool operator!=(const InputSignature& other) const {
    return !(*this == other);
  }

  bool empty() const { return ptrs.empty(); }
};

// Helper to create input signature from tensor list
inline InputSignature make_input_signature(const std::vector<torch::Tensor>& tensors) {
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

// Helper to create input signature from multiple tensor lists (for training)
inline InputSignature make_input_signature(const std::vector<torch::Tensor>& inputs,
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

// Validate that current inputs match the captured signature
inline void validate_input_signature(const InputSignature& expected,
                                     const InputSignature& actual,
                                     const char* context) {
  if (expected != actual) {
    std::stringstream ss;
    ss << "CUDA graph input mismatch in " << context << ". "
       << "When cuda_graphs is enabled, input tensors must have consistent "
       << "data pointers, shapes, and dtypes across all calls. "
       << "If you need to change inputs, disable cuda_graphs.";
    THROW_INVALID_USAGE(ss.str());
  }
}

// Graph state for inference
struct InferenceGraphState {
  int warmup_count = 0;
  bool captured = false;

  InputSignature input_signature;
  CudaGraph graph;
  CudaGraphExec graph_exec;
  std::vector<torch::Tensor> static_outputs;
};

// Graph state for training (single graph for forward + loss + backward)
struct TrainingGraphState {
  int warmup_count = 0;
  bool captured = false;

  InputSignature input_signature;
  CudaGraph graph;
  CudaGraphExec graph_exec;
  torch::Tensor static_loss;
};

// Graph state for a model, including the capture stream
class ModelGraphState {
public:
  InferenceGraphState inference;
  TrainingGraphState training;

  ModelGraphState(int device_index = 0)
      : capture_stream_(nullptr), device_index_(device_index) {
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

  // Get c10 stream wrapper for the capture stream (for PyTorch integration)
  c10::cuda::CUDAStream get_capture_cuda_stream() const {
    return c10::cuda::getStreamFromExternal(capture_stream_, device_index_);
  }

private:
  cudaStream_t capture_stream_;
  int device_index_;
};

} // namespace torchfort

#endif // ENABLE_GPU

