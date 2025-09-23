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

#include <mpi.h>
#ifdef ENABLE_GPU
#include <nccl.h>

#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/distributed.h"

namespace torchfort {

static MPI_Datatype get_mpi_dtype(torch::Tensor tensor) {
  auto dtype = tensor.dtype();

  if (dtype == torch::kFloat32) {
    return MPI_FLOAT;
  } else if (dtype == torch::kFloat64) {
    return MPI_DOUBLE;
  } else if (dtype == torch::kLong) {
    return MPI_INT64_T;
  } else {
    THROW_INVALID_USAGE("Unsupported dtype encountered.");
  }
}

#ifdef ENABLE_GPU
static ncclDataType_t get_nccl_dtype(torch::Tensor tensor) {
  auto dtype = tensor.dtype();

  if (dtype == torch::kFloat32) {
    return ncclFloat;
  } else if (dtype == torch::kFloat64) {
    return ncclDouble;
  } else if (dtype == torch::kLong) {
    return ncclInt64;
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
#endif

void Comm::initialize(bool initialize_nccl) {
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &size));

#ifdef ENABLE_GPU
  if (initialize_nccl) {
    nccl_comm = ncclCommFromMPIComm(mpi_comm);

    int greatest_priority;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
    CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));

    CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
#endif

  initialized = true;
}

void Comm::finalize() {
#ifdef ENABLE_GPU
  if (nccl_comm)
    CHECK_NCCL(ncclCommDestroy(nccl_comm));
  if (stream)
    CHECK_CUDA(cudaStreamDestroy(stream));
  if (event)
    CHECK_CUDA(cudaEventDestroy(event));
#endif
}

void Comm::allreduce(torch::Tensor& tensor, bool average) const {

  if (!tensor.is_contiguous()) {
    THROW_NOT_SUPPORTED("allreduce method does not support non-contiguous tensors.");
  }

  auto dtype = tensor.dtype();
  if ((dtype == torch::kLong) && average) {
    THROW_NOT_SUPPORTED("allreduce method for integer-valued tensors does not support averaging.");
  }

#ifdef ENABLE_GPU
  if (tensor.device().type() == torch::kCUDA) {
    auto torch_stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDA(cudaEventRecord(event, torch_stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream, event));

    auto count = torch::numel(tensor);
    ncclDataType_t nccl_dtype;
    if (torch::is_complex(tensor)) {
      nccl_dtype = get_nccl_dtype(torch::view_as_real(tensor));
      count *= 2;
    } else {
      nccl_dtype = get_nccl_dtype(tensor);
    }
    CHECK_NCCL(ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), count, nccl_dtype, (average) ? ncclAvg : ncclSum,
                             nccl_comm, stream));

    CHECK_CUDA(cudaEventRecord(event, stream));
    CHECK_CUDA(cudaStreamWaitEvent(torch_stream, event));
  } else if (tensor.device().type() == torch::kCPU) {
#endif
    auto count = torch::numel(tensor);
    MPI_Datatype mpi_dtype;
    if (torch::is_complex(tensor)) {
      mpi_dtype = get_mpi_dtype(torch::view_as_real(tensor));
      count *= 2;
    } else {
      mpi_dtype = get_mpi_dtype(tensor);
    }
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), count, mpi_dtype, MPI_SUM, mpi_comm));

    if (average) {
      tensor /= size;
    }
#ifdef ENABLE_GPU
  }
#endif
}

void Comm::allreduce(std::vector<torch::Tensor>& tensors, bool average) const {

#ifdef ENABLE_GPU
  if (tensors[0].device().type() == torch::kCUDA) {
    auto torch_stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDA(cudaEventRecord(event, torch_stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream, event));
    CHECK_NCCL(ncclGroupStart());
  }
#endif

  for (auto& t : tensors) {
    if (!t.is_contiguous()) {
      THROW_NOT_SUPPORTED("allreduce method does not support non-contiguous tensors.");
    }
    allreduce(t, average);
  }

#ifdef ENABLE_GPU
  if (tensors[0].device().type() == torch::kCUDA) {
    CHECK_NCCL(ncclGroupEnd());
    auto torch_stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDA(cudaEventRecord(event, stream));
    CHECK_CUDA(cudaStreamWaitEvent(torch_stream, event));
  }
#endif
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
  if (!tensor.is_contiguous()) {
    THROW_NOT_SUPPORTED("broadcast method does not support non-contiguous tensors.");
  }

  auto count = torch::numel(tensor);
#ifdef ENABLE_GPU
  if (tensor.device().type() == torch::kCUDA) {
    // Use NCCL for GPU tensors
    ncclDataType_t nccl_dtype;
    if (torch::is_complex(tensor)) {
      nccl_dtype = get_nccl_dtype(torch::view_as_real(tensor));
      count *= 2;
    } else {
      nccl_dtype = get_nccl_dtype(tensor);
    }

    auto torch_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
    CHECK_CUDA(cudaEventRecord(event, torch_stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream, event));

    CHECK_NCCL(ncclBroadcast(tensor.data_ptr(), tensor.data_ptr(), count, nccl_dtype, root, nccl_comm, stream));

    CHECK_CUDA(cudaEventRecord(event, stream));
    CHECK_CUDA(cudaStreamWaitEvent(torch_stream, event));
  } else if (tensor.device().type() == torch::kCPU) {
#endif
    // Use MPI for CPU tensors
    MPI_Datatype mpi_dtype;
    if (torch::is_complex(tensor)) {
      mpi_dtype = get_mpi_dtype(torch::view_as_real(tensor));
      count *= 2;
    } else {
      mpi_dtype = get_mpi_dtype(tensor);
    }

    CHECK_MPI(MPI_Bcast(tensor.data_ptr(), count, mpi_dtype, root, mpi_comm));

#ifdef ENABLE_GPU
  }
#endif
}

} // namespace torchfort
