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
