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
