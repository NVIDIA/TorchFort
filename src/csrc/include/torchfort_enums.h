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

#define TORCHFORT_DEVICE_CPU (-1)

/**
 * @brief This enum defines the data types supported.
 */
enum torchfort_datatype_t { TORCHFORT_FLOAT = -1, TORCHFORT_DOUBLE = -2, TORCHFORT_INT32 = -3, TORCHFORT_INT64 = -4 };

/**
 * @brief This enum defines the possible values return values from TorchFort. Most functions in the TorchFort library
 * will return one of these values to indicate if an operation has completed successfully or an error occured.
 */
enum torchfort_result_t {
  TORCHFORT_RESULT_SUCCESS = 0,        ///< The operation completed successfully
  TORCHFORT_RESULT_INVALID_USAGE = 1,  ///< A user error, typically an invalid argument
  TORCHFORT_RESULT_NOT_SUPPORTED = 2,  ///< A user error, requesting an invalid or unsupported operation configuration
  TORCHFORT_RESULT_INTERNAL_ERROR = 3, ///< An internal library error, should be reported
  TORCHFORT_RESULT_CUDA_ERROR = 4,     ///< An error occured in the CUDA Runtime
  TORCHFORT_RESULT_MPI_ERROR = 5,      ///< An error occured in the MPI library
  TORCHFORT_RESULT_NCCL_ERROR = 6      ///< An error occured in the NCCL library
};
