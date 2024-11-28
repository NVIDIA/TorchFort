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
#include <functional>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "internal/base_loss.h"
#include "internal/base_model.h"
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
