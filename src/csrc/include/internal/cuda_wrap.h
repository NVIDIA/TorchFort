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

#pragma once

#include <mutex>
#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#endif

#define DECLARE_CUDA_PFN(symbol, version) PFN_##symbol##_v##version pfn_##symbol = nullptr

namespace torchfort {

struct cuFunctionTable {
#if CUDART_VERSION >= 11030
  DECLARE_CUDA_PFN(cuCtxGetCurrent, 4000);
  DECLARE_CUDA_PFN(cuCtxGetDevice, 2000);
  DECLARE_CUDA_PFN(cuCtxSetCurrent, 4000);
  DECLARE_CUDA_PFN(cuGetErrorString, 6000);
  DECLARE_CUDA_PFN(cuStreamGetCtx, 9020);
#if CUDART_VERSION >= 12080
  DECLARE_CUDA_PFN(cuStreamGetDevice, 12080);
#endif
#endif
  bool initialized = false;
  std::mutex mutex;
};

extern cuFunctionTable cuFnTable;

void initCuFunctionTable();
} // namespace torchfort

#undef DECLARE_CUDA_PFN
