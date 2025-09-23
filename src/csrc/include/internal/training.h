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
#endif

#include <torch/torch.h>

#include <internal/tensor_list.h>

namespace torchfort {

void inference_multiarg(const char* name, torchfort_tensor_list_t inputs_in, torchfort_tensor_list_t outputs_in,
                        cudaStream_t ext_stream = 0);

void train_multiarg(const char* name, torchfort_tensor_list_t inputs_in, torchfort_tensor_list_t labels_in,
                    float* loss_val, torchfort_tensor_list_t extra_loss_args_in, cudaStream_t ext_stream = 0);

template <MemoryLayout L, typename T>
void inference(const char* name, T* input, size_t input_dim, int64_t* input_shape, T* output, size_t output_dim,
               int64_t* output_shape, cudaStream_t ext_stream = 0) {
  TensorList inputs, outputs;

  inputs.add_tensor<L>(input, input_dim, input_shape);
  outputs.add_tensor<L>(output, output_dim, output_shape);

  inference_multiarg(name, &inputs, &outputs, ext_stream);
}

template <MemoryLayout L, typename T>
void train(const char* name, T* input, size_t input_dim, int64_t* input_shape, T* label, size_t label_dim,
           int64_t* label_shape, T* loss_val, cudaStream_t ext_stream = 0) {
  TensorList inputs, labels;

  inputs.add_tensor<L>(input, input_dim, input_shape);
  labels.add_tensor<L>(label, label_dim, label_shape);

  // multiarg API expects float loss value, so use temporary here
  float loss_val_tmp;
  train_multiarg(name, &inputs, &labels, &loss_val_tmp, nullptr, ext_stream);
  *loss_val = loss_val_tmp;
}

} // namespace torchfort
