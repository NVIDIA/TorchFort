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
#include <functional>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "internal/base_loss.h"
#include "internal/base_model.h"
#include "internal/cuda_wrap.h"
#include "internal/exceptions.h"
#include "internal/utils.h"

#define CHECK_TORCHFORT(call)                                                                                          \
  do {                                                                                                                 \
    torchfort_result_t err = call;                                                                                     \
    if (TORCHFORT_RESULT_SUCCESS != err) {                                                                             \
      std::ostringstream os;                                                                                           \
      os << "error code " << err;                                                                                      \
      throw torchfort::InternalError(__FILE__, __LINE__, os.str().c_str());                                            \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUDA(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (cudaSuccess != err) {                                                                                          \
      throw torchfort::CudaError(__FILE__, __LINE__, cudaGetErrorString(err));                                         \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUDA_DRV(call)                                                                                           \
  do {                                                                                                                 \
    if (!cuFnTable.initialized) {initCuFunctionTable();}                                                               \
    CUresult err = cuFnTable.pfn_##call;                                                                               \
    if (CUDA_SUCCESS != err) {                                                                                         \
      const char* error_str;                                                                                           \
      cuFnTable.pfn_cuGetErrorString(err, &error_str);                                                                 \
      throw torchfort::CudaError(__FILE__, __LINE__, error_str);                                                       \
    }                                                                                                                  \
  } while (false)

#define CHECK_NCCL(call)                                                                                               \
  do {                                                                                                                 \
    ncclResult_t err = call;                                                                                           \
    if (ncclSuccess != err) {                                                                                          \
      std::ostringstream os;                                                                                           \
      os << "error code " << err;                                                                                      \
      throw torchfort::NcclError(__FILE__, __LINE__, os.str().c_str());                                                \
    }                                                                                                                  \
  } while (false)

#define CHECK_MPI(call)                                                                                                \
  do {                                                                                                                 \
    int err = call;                                                                                                    \
    if (0 != err) {                                                                                                    \
      char error_str[MPI_MAX_ERROR_STRING];                                                                            \
      int len;                                                                                                         \
      MPI_Error_string(err, error_str, &len);                                                                          \
      if (error_str) {                                                                                                 \
        throw torchfort::MpiError(__FILE__, __LINE__, error_str);                                                      \
      } else {                                                                                                         \
        std::ostringstream os;                                                                                         \
        os << "error code " << err;                                                                                    \
        throw torchfort::MpiError(__FILE__, __LINE__, os.str().c_str());                                               \
      }                                                                                                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

#define IS_CUDA_DRV_FUNC_AVAILABLE(symbol)                                                                             \
  ([&]() { if (!cuFnTable.initialized) {initCuFunctionTable();}                                                        \
    return cuFnTable.pfn_##symbol != nullptr;                                                                          \
  })()


#define BEGIN_MODEL_REGISTRY                                                                                           \
  static std::unordered_map<std::string, std::function<std::shared_ptr<BaseModel>()>> model_registry {

#define REGISTER_MODEL(name, cls) {sanitize(#name), [] { return std::shared_ptr<BaseModel>(new cls()); }},

#define END_MODEL_REGISTRY                                                                                             \
  }                                                                                                                    \
  ;

#define BEGIN_LOSS_REGISTRY                                                                                            \
  static std::unordered_map<std::string, std::function<std::shared_ptr<BaseLoss>()>> loss_registry {

#define REGISTER_LOSS(name, cls) {sanitize(#name), [] { return std::shared_ptr<BaseLoss>(new cls()); }},

#define END_LOSS_REGISTRY                                                                                              \
  }                                                                                                                    \
  ;
