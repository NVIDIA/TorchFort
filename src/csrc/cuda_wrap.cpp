/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef ENABLE_GPU
#include <cuda_runtime.h>

#include "internal/cuda_wrap.h"
#include "internal/defines.h"

#if CUDART_VERSION >= 13000
#define LOAD_SYM(symbol, version, optional)                                                                   \
  do {                                                                                                        \
    cudaDriverEntryPointQueryResult driverStatus = cudaDriverEntryPointSymbolNotFound;                        \
    cudaError_t err = cudaGetDriverEntryPointByVersion(#symbol, (void**)(&cuFnTable.pfn_##symbol), version,   \
                                                       cudaEnableDefault, &driverStatus));                    \
    if ((driverStatus != cudaDriverEntryPointSuccess || err != cudaSuccess) && !optional) {                   \
      THROW_CUDA_ERROR("cudaGetDriverEntryPointByVersion failed.");                                           \
    }                                                                                                         \
  } while (false)
#elif CUDART_VERSION >= 12000
#define LOAD_SYM(symbol, version, optional)                                                                   \
  do {                                                                                                        \
    cudaDriverEntryPointQueryResult driverStatus = cudaDriverEntryPointSymbolNotFound;                        \
    cudaError_t err = cudaGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), cudaEnableDefault,  \
                                              &driverStatus);                                                 \
    if ((driverStatus != cudaDriverEntryPointSuccess || err != cudaSuccess) && !optional) {                   \
      THROW_CUDA_ERROR("cudaGetDriverEntryPoint failed.");                                                    \
    }                                                                                                         \
  } while (false)
#else
#define LOAD_SYM(symbol, version, optional)                                                                   \
  do {                                                                                                        \
    cudaError_t err = cudaGetDriverEntryPoint(#symbol, (void**)(&cuFnTable.pfn_##symbol), cudaEnableDefault); \
    if (err != cudaSuccess && !optional) {                                                                    \
      THROW_CUDA_ERROR("cudaGetDriverEntryPoint failed.");                                                    \
    }                                                                                                         \
  } while (false)
#endif

namespace torchfort {

cuFunctionTable cuFnTable; // global table of required CUDA driver functions

void initCuFunctionTable() {
  std::lock_guard<std::mutex> guard(cuFnTable.mutex);

  if (cuFnTable.initialized) {
    return;
  }

#if CUDART_VERSION >= 11030
  LOAD_SYM(cuCtxGetCurrent, 4000, false);
  LOAD_SYM(cuCtxGetDevice, 2000, false);
  LOAD_SYM(cuCtxSetCurrent, 4000, false);
  LOAD_SYM(cuGetErrorString, 6000, false);
  LOAD_SYM(cuStreamGetCtx, 9020, false);
#if CUDART_VERSION >= 12080
  LOAD_SYM(cuStreamGetDevice, 12080, true);
#endif
#endif
  cuFnTable.initialized = true;
}

} // namespace torchfort

#undef LOAD_SYM
#endif
