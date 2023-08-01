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
