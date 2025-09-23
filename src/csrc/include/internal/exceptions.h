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

#include <exception>
#include <iostream>
#include <string>

#include "torchfort.h"

// Useful defines for throwing with line/file info
#define THROW_INVALID_USAGE(msg)                                                                                       \
  do {                                                                                                                 \
    throw torchfort::InvalidUsage(__FILE__, __LINE__, msg);                                                            \
  } while (false)
#define THROW_NOT_SUPPORTED(msg)                                                                                       \
  do {                                                                                                                 \
    throw torchfort::NotSupported(__FILE__, __LINE__, msg);                                                            \
  } while (false)
#define THROW_INTERNAL_ERROR(msg)                                                                                      \
  do {                                                                                                                 \
    throw torchfort::InternalError(__FILE__, __LINE__, msg);                                                           \
  } while (false)
#define THROW_CUDA_ERROR(msg)                                                                                          \
  do {                                                                                                                 \
    throw torchfort::CudaError(__FILE__, __LINE__, msg);                                                               \
  } while (false)
#define THROW_MPI_ERROR(msg)                                                                                           \
  do {                                                                                                                 \
    throw torchfort::MpiError(__FILE__, __LINE__, msg);                                                                \
  } while (false)
#define THROW_NCCL_ERROR(msg)                                                                                          \
  do {                                                                                                                 \
    throw torchfort::NcclError(__FILE__, __LINE__, msg);                                                               \
  } while (false)
#define THROW_NVSHMEM_ERROR(msg)                                                                                       \
  do {                                                                                                                 \
    throw torchfort::NvshmemError(__FILE__, __LINE__, msg);                                                            \
  } while (false)

namespace torchfort {

class BaseException : public std::exception {
public:
  BaseException(const std::string& file, int line, const std::string& generic_info,
                const std::string& extra_info = std::string()) {
    s = "TORCHFORT:ERROR: ";
    s += file + ":" + std::to_string(line) + " ";
    s += generic_info;
    if (!extra_info.empty()) {
      s += " (" + extra_info + ")\n";
    } else {
      s += "\n";
    }
  }

  const char* what() const throw() { return s.c_str(); }

  virtual torchfort_result_t getResult() const = 0;

private:
  std::string s;
};

class InvalidUsage : public BaseException {
public:
  InvalidUsage(const std::string& file, int line, const std::string& extra_info = std::string())
      : BaseException(file, line, "Invalid usage.", extra_info){};
  torchfort_result_t getResult() const override { return TORCHFORT_RESULT_INVALID_USAGE; }
};

class NotSupported : public BaseException {
public:
  NotSupported(const std::string& file, int line, const std::string& extra_info = std::string())
      : BaseException(file, line, "Not supported.", extra_info){};
  torchfort_result_t getResult() const override { return TORCHFORT_RESULT_NOT_SUPPORTED; }
};

class InternalError : public BaseException {
public:
  InternalError(const std::string& file, int line, const std::string& extra_info = std::string())
      : BaseException(file, line, "Internal error.", extra_info){};
  torchfort_result_t getResult() const override { return TORCHFORT_RESULT_INTERNAL_ERROR; }
};

class CudaError : public BaseException {
public:
  CudaError(const std::string& file, int line, const std::string& extra_info = std::string())
      : BaseException(file, line, "CUDA error.", extra_info){};
  torchfort_result_t getResult() const override { return TORCHFORT_RESULT_CUDA_ERROR; }
};

class MpiError : public BaseException {
public:
  MpiError(const std::string& file, int line, const std::string& extra_info = std::string())
      : BaseException(file, line, "MPI error.", extra_info){};
  torchfort_result_t getResult() const override { return TORCHFORT_RESULT_MPI_ERROR; }
};

class NcclError : public BaseException {
public:
  NcclError(const std::string& file, int line, const std::string& extra_info = std::string())
      : BaseException(file, line, "NCCL error.", extra_info){};
  torchfort_result_t getResult() const override { return TORCHFORT_RESULT_NCCL_ERROR; }
};

} // namespace torchfort
