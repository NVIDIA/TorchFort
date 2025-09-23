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
#include <nccl.h>
#endif
#include <mpi.h>

#include <torch/torch.h>

namespace torchfort {

struct Comm {
  void initialize(bool initialize_nccl = false);
  void finalize();
  void allreduce(torch::Tensor& tensor, bool average = false) const;
  void allreduce(std::vector<torch::Tensor>& tensors, bool average = false) const;
  void allreduce(double& val, bool average = false) const;
  void allreduce(float& val, bool average = false) const;
  void broadcast(torch::Tensor& tensor, int root = 0) const;

  int rank;
  int size;
  MPI_Comm mpi_comm;
#ifdef ENABLE_GPU
  ncclComm_t nccl_comm = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
#endif
  bool initialized = false;

  Comm(MPI_Comm mpi_comm) : mpi_comm(mpi_comm){};
};

} // namespace torchfort
