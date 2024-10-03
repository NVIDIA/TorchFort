/*
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdint.h>
#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif
#include <mpi.h>

#include "torchfort_enums.h"
#include "torchfort_rl.h"

#define TORCHFORT_MAJOR 0
#define TORCHFORT_MINOR 2
#define TORCHFORT_PATCH 0

#define WANDB_LOG_FUNC(dtype)                                                                                          \
  torchfort_result_t torchfort_wandb_log_##dtype(const char* name, const char* metric_name, int64_t step,              \
                                                 dtype value) {                                                        \
    torchfort::wandb_log(name, metric_name, step, value);                                                              \
    return TORCHFORT_RESULT_SUCCESS;                                                                                   \
  }

#define WANDB_LOG_PROTO(dtype)                                                                                         \
  torchfort_result_t torchfort_wandb_log_##dtype(const char* name, const char* metric_name, int64_t step, dtype value);

/**
 * @brief A pointer to a TorchFort tensor list object
 */
typedef void* torchfort_tensor_list_t;


#ifdef __cplusplus
extern "C" {
#endif

// Model creation functions
/**
 * @brief Creates a model instance from a provided configuration file.
 *
 * @param[in] name A name to assign to the created model instance to use as a key for other TorchFort routines.
 * @param[in] config_fname The filesystem path to the user-defined model configuration file to use.
 * @param[in] device Which device to place and run the model on. For TORCHFORT_DEVICE_CPU (-1), model will be placed on
 * CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_create_model(const char* name, const char* config_fname, int device);
/**
 * @brief Creates a distributed data-parallel model from a provided configuration file.
 *
 * @param[in] name A name to assign to created model to use as a key for other TorchFort routines.
 * @param[in] config_fname The filesystem path to the user-defined model configuration file to use.
 * @param[in] mpi_comm MPI communicator to use to initialize NCCL communication library for data-parallel communication.
 * @param[in] device Which device to place and run the model on. For TORCHFORT_DEVICE_CPU (-1), model will be placed on
 * CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_create_distributed_model(const char* name, const char* config_fname, MPI_Comm mpi_comm,
                                                      int device);

// Training and inference functions
/**
 * @brief Runs a training iteration of a model instance using provided input and label data.
 *
 * @param[in] name The name of model instance to use, as defined during model creation.
 * @param[in] input A pointer to a memory buffer containing input data.
 * @param[in] input_dim Rank of the input data.
 * @param[in] input_shape A pointer to an array specifying the shape of the input data. Length should be equal to the
 * rank of the input data.
 * @param[in] label A pointer to a memory buffer containing label data.
 * @param[in] label_dim Rank of the label data.
 * @param[in] label_shape A pointer to an array specifying the shape of the label data. Length should be equal to the
 * rank of the label data.
 * @param[out] loss_val A pointer to a memory location to write the loss value computed during the training iteration.
 * @param[in] dtype The TorchFort datatype to use for this operation.
 * @param[in] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_train(const char* name, void* input, size_t input_dim, int64_t* input_shape, void* label,
                                   size_t label_dim, int64_t* label_shape, void* loss_val, torchfort_datatype_t dtype,
                                   cudaStream_t stream);

torchfort_result_t torchfort_train_F(const char* name, void* input, size_t input_dim, int64_t* input_shape, void* label,
                                     size_t label_dim, int64_t* label_shape, void* loss_val, torchfort_datatype_t dtype,
                                     cudaStream_t stream);

/**
 * @brief Runs a training iteration of a model instance using provided input and label tensor lists.
 *
 * @param[in] name The name of model instance to use, as defined during model creation.
 * @param[in] inputs A tensor list of input tensors.
 * @param[in] labels A tensor list of label tensors.
 * @param[out] loss_val A pointer to a single precision scalar to write the loss value computed during the training iteration.
 * @param[in] extra_loss_args A tensor list of additional tensors to pass into loss computation. Set to nullptr if unused.
 * @param[in] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_train_multiarg(const char* name, torchfort_tensor_list_t inputs, torchfort_tensor_list_t labels,
                                            float* loss_val, torchfort_tensor_list_t extra_loss_args, cudaStream_t stream);

/**
 * @brief Runs inference on a model using provided input data.
 *
 * @param[in] name The name of model instance to use, as defined during model creation.
 * @param[in] input A pointer to a memory buffer containing input data.
 * @param[in] input_dim Rank of the input data.
 * @param[in] input_shape A pointer to an array specifying the shape of the input data. Length should be equal to the
 * rank of the input data.
 * @param[in,out] output A pointer to a memory buffer to write output data.
 * @param[in] output_dim Rank of the output data.
 * @param[in] output_shape  A pointer to an array specifying the shape of the output data. Length should be equal to the
 * rank of the output data.
 * @param[in] dtype The TorchFort datatype to use for this operation.
 * @param[in] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_inference(const char* name, void* input, size_t input_dim, int64_t* input_shape,
                                       void* output, size_t output_dim, int64_t* output_shape,
                                       torchfort_datatype_t dtype, cudaStream_t stream);

torchfort_result_t torchfort_inference_F(const char* name, void* input, size_t input_dim, int64_t* input_shape,
                                         void* output, size_t output_dim, int64_t* output_shape,
                                         torchfort_datatype_t dtype, cudaStream_t stream);

/**
 * @brief Runs inference on a model using provided input tensor list.
 *
 * @param[in] name The name of model instance to use, as defined during model creation.
 * @param[in] inputs A tensor list of input tensors.
 * @param[in,out] outputs A tensor list of output tensors.
 * @param[in] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_inference_multiarg(const char* name, torchfort_tensor_list_t inputs,
                                                torchfort_tensor_list_t outputs, cudaStream_t stream);

// Model/Checkpoint save and loading functions
/**
 * @brief Saves a model to file.
 *
 * @param[in] name The name of model instance to save, as defined during model creation.
 * @param[in] fname The filename to save the model weights to.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_save_model(const char* name, const char* fname);

/**
 * @brief Loads a model from a file.
 *
 * @param[in] name The name of model instance to load the model weights to, as defined during model creation.
 * @param[in] fname The filename to load the model from.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_load_model(const char* name, const char* fname);

/**
 * @brief Saves a training checkpoint to a directory. In contrast to \p torchfort_save_model, this function saves
 * additional state to restart training, like the optimizer states and learning rate schedule progress.
 *
 * @param[in] name The name of model instance to save, as defined during model creation.
 * @param[in] checkpoint_dir A writeable filesystem path to a directory to save the checkpoint data to.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_save_checkpoint(const char* name, const char* checkpoint_dir);

/**
 * @brief Loads a training checkpoint from a directory. In contrast to the \p torchfort_load_model, this function loads
 * additional state to restart training, like the optimizer states and learning rate schedule progress.
 *
 * @param[in] name The name of model instance to load checkpoint data into, as defined during model creation.
 * @param[in] checkpoint_dir A readable filesystem path to a directory to load the checkpoint data from.
 * @param[out] step_train A pointer to an integer to write current training step for loaded checkpoint.
 * @param[out] step_inference A pointer to an integer to write current inference step for loaded checkpoint.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_load_checkpoint(const char* name, const char* checkpoint_dir, int64_t* step_train,
                                             int64_t* step_inference);

// Miscellaneous utility functions
/**
 * @brief Utility function to enable/disable cuDNN runtime autotuning in PyTorch.
 * @param[in] flag Boolean value to set the cuDNN benchmarking flag to.
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_set_cudnn_benchmark(const bool flag);

/**
 * @brief Utility function to enable/disable TF32 support in PyTorch.
 * @param[in] flag Boolean value to set the TF32 flag to.
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_set_cuda_allow_tf32(const bool flag);

/**
 * @brief Utility function to set a seed for the host in PyTorch.
 * @param[in] seed An integer value to be used as seed for any host RNG.
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_set_manual_seed(const int seed);

/**
 * @brief Utility function to set a seed for cuda devices in PyTorch.
 * @param[in] seed An integer value to be used as seed for any device RNG.
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_set_cuda_manual_seed(const int seed);

// Weights and Bias Logging functions
/**
 * @brief Write an integer value to a Weights and Bias log. Use the \p _float and  \p _double variants to write \p float
 * and \p double values respectively.
 *
 * @param[in] name The name of model instance to associate this metric value with, as defined during model creation.
 * @param[in] metric_name Metric label.
 * @param[in] step Training/inference step to associate with metric value.
 * @param[in] value Metric value to log.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
WANDB_LOG_PROTO(int)
WANDB_LOG_PROTO(float)
WANDB_LOG_PROTO(double)


// Tensor List Management functions
/**
 * @brief Creates a TorchFort tensor list.
 *
 * @param[out] tensor_list A pointer to an uninitialized torchfort_tensor_list_t
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_tensor_list_create(torchfort_tensor_list_t* tensor_list);

/**
 * @brief Destroys a TorchFort tensor list.
 *
 * @param[in] tensor_list A TorchFort tensor list
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_tensor_list_destroy(torchfort_tensor_list_t tensor_list);

/**
 * @brief Adds a tensor to a TorchFort tensor list. Tensor data is added by reference, so changes to externally
 * provided memory will modify tensors contained in the list.
 *
 * @param[in] tensor_list A TorchFort tensor list
 * @param[in] data_ptr A pointer to a memory buffer containing tensor data.
 * @param[in] dim Rank of the tensor data.
 * @param[in] shape A pointer to an array specifying the shape of the tensor data. Length should be equal to the
 * rank of the tensor data.
 * @param[in] dtype The TorchFort datatype of the tensor data.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_tensor_list_add_tensor(torchfort_tensor_list_t tensor_list, void* data_ptr, size_t dim,
                                                    int64_t* shape, torchfort_datatype_t dtype);
torchfort_result_t torchfort_tensor_list_add_tensor_F(torchfort_tensor_list_t tensor_list, void* data_ptr, size_t dim,
                                                      int64_t* shape, torchfort_datatype_t dtype);


#ifdef __cplusplus
}
#endif
