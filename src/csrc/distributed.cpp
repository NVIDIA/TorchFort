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

#include <mpi.h>
#include <nccl.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/distributed.h"

namespace torchfort {

static ncclDataType_t get_nccl_dtype(torch::Tensor tensor) {
  auto dtype = tensor.dtype();

  if (dtype == torch::kFloat32) {
    return ncclFloat;
  } else if (dtype == torch::kFloat64) {
    return ncclDouble;
  } else {
    THROW_INVALID_USAGE("Unsupported dtype encountered.");
  }
}

static ncclComm_t ncclCommFromMPIComm(MPI_Comm mpi_comm) {
  int rank, size;
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &size));

  ncclUniqueId id;
  if (rank == 0)
    CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm));
  ncclComm_t nccl_comm;
  CHECK_NCCL(ncclCommInitRank(&nccl_comm, size, id, rank));

  return nccl_comm;
}

void Comm::initialize(MPI_Comm mpi_comm_) {
  mpi_comm = mpi_comm_;
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &size));

  nccl_comm = ncclCommFromMPIComm(mpi_comm);

  int greatest_priority;
  CHECK_CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
  CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));

  CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
}

void Comm::finalize() {
  CHECK_NCCL(ncclCommDestroy(nccl_comm));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaEventDestroy(event));
}

void Comm::allreduce(torch::Tensor& tensor, bool average) const {
  if (tensor.device().type() != torch::kCUDA) {
    THROW_INVALID_USAGE("allreduce only supports GPU tensors for now.");
  }
  auto torch_stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDA(cudaEventRecord(event, torch_stream));
  CHECK_CUDA(cudaStreamWaitEvent(stream, event));

  auto count = torch::numel(tensor);
  CHECK_NCCL(ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), count, get_nccl_dtype(tensor),
                           (average) ? ncclAvg : ncclSum, nccl_comm, stream));

  CHECK_CUDA(cudaEventRecord(event, stream));
  CHECK_CUDA(cudaStreamWaitEvent(torch_stream, event));
}

void Comm::allreduce(const std::vector<torch::Tensor>& tensors, bool average) const {
  auto torch_stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDA(cudaEventRecord(event, torch_stream));
  CHECK_CUDA(cudaStreamWaitEvent(stream, event));

  CHECK_NCCL(ncclGroupStart());
  for (auto& t : tensors) {
    if (t.device().type() != torch::kCUDA) {
      THROW_INVALID_USAGE("allreduce only supports GPU tensors for now.");
    }

    auto count = torch::numel(t);
    CHECK_NCCL(ncclAllReduce(t.data_ptr(), t.data_ptr(), count, get_nccl_dtype(t), (average) ? ncclAvg : ncclSum,
                             nccl_comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  CHECK_CUDA(cudaEventRecord(event, stream));
  CHECK_CUDA(cudaStreamWaitEvent(torch_stream, event));
}
void Comm::allreduce(double& val, bool average) const {
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, mpi_comm));
  if (average) {
    val /= size;
  }
}
void Comm::allreduce(float& val, bool average) const {
  CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_FLOAT, MPI_SUM, mpi_comm));
  if (average) {
    val /= size;
  }
}

void Comm::broadcast(torch::Tensor& tensor, int root) const {
  if (tensor.device().type() != torch::kCUDA) {
    THROW_INVALID_USAGE("broadcast only supports GPU tensors for now.");
  }
  auto count = torch::numel(tensor);

  auto torch_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
  CHECK_CUDA(cudaEventRecord(event, torch_stream));
  CHECK_CUDA(cudaStreamWaitEvent(stream, event));

  CHECK_NCCL(
      ncclBroadcast(tensor.data_ptr(), tensor.data_ptr(), count, get_nccl_dtype(tensor), root, nccl_comm, stream));

  CHECK_CUDA(cudaEventRecord(event, stream));
  CHECK_CUDA(cudaStreamWaitEvent(torch_stream, event));
}

} // namespace torchfort
