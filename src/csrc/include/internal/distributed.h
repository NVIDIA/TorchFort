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
