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
#include "torchfort_enums.h"
#include <cuda_runtime.h>

#define RL_WANDB_LOG_FUNC(dtype)                                                                                       \
  torchfort_result_t torchfort_rl_off_policy_wandb_log_##dtype(const char* name, const char* metric_name,              \
                                                               int64_t step, dtype value) {                            \
    torchfort::rl::wandb_log_system(name, metric_name, step, value);                                                   \
    return TORCHFORT_RESULT_SUCCESS;                                                                                   \
  }

#define RL_WANDB_LOG_PROTO(dtype)                                                                                      \
  torchfort_result_t torchfort_rl_off_policy_wandb_log_##dtype(const char* name, const char* metric_name,              \
                                                               int64_t step, dtype value);

#ifdef __cplusplus
extern "C" {
#endif

// RL system creation functions
/**
 * @brief Creates an off-policy reinforcement learning training system instance from a provided configuration file.
 *
 * @param[in] name A name to assign to the created training system instance to use as a key for other TorchFort
 * routines.
 * @param[in] config_fname The filesystem path to the user-defined configuration file to use.
 * @param[in] model device Which device to place and run the model on. For TORCHFORT_DEVICE_CPU (-1), model will be placed on CPU. For 
 * values >= 0, model will be placed on GPU with index corresponding to value.
 * @param[in] rb_device Which device to place the replay buffer on. For TORCHFORT_DEVICE_CPU (-1), the replay buffer will be placed on CPU. For 
 * values >= 0, the replay buffer will be placed on GPU with index corresponding to value.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_create_system(const char* name, const char* config_fname,
							 int model_device, int rb_device);

/**
 * @brief Creates a (synchronous) data-parallel off-policy reinforcement learning system instance from a provided
 * configuration file.
 *
 * @param[in] name A name to assign to the created training system instance to use as a key for other TorchFort
 * routines.
 * @param[in] config_fname The filesystem path to the user-defined configuration file to use.
 * @param[in] mpi_comm MPI communicator to use to initialize NCCL communication library for data-parallel communication.
 * @param[in] model device Which device to place and run the model on. For TORCHFORT_DEVICE_CPU (-1), model will be placed on CPU. For 
 * values >= 0, model will be placed on GPU with index corresponding to value.
 * @param[in] rb_device Which device to place the replay buffer on. For TORCHFORT_DEVICE_CPU (-1), the replay buffer will be placed on CPU. For 
 * values >= 0, the replay buffer will be placed on GPU with index corresponding to value.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_create_distributed_system(const char* name, const char* config_fname, MPI_Comm mpi_comm,
								     int model_device, int rb_device);

//  RL training/prediction/evaluation functions
/**
 * @brief Runs a training iteration of an off-policy refinforcement learning instance and returns loss values for policy
 and value functions
 * @details This routine samples a batch of specified size from the replay buffer according to the buffers sampling
 procedure
 * and performs a train step using this sample. The details of the training procedure are abstracted away from the user
 and depend on the
 * chosen system algorithm.
 *
 * @param[in] name The name of system instance to use, as defined during system creation.
 * @param[out] p_loss_val A pointer to a memory location to write the policy loss value computed during the training
 iteration.
 * @param[out] q_loss_val A pointer to a memory location to write the critic loss value computed during the training
 iteration. If the system uses multiple critics, the average across all critics is returned.
 * @param[out] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_train_step(const char* name, float* p_loss_val, float* q_loss_val,
                                                      cudaStream_t stream);

/**
 * @brief Suggests an action based on the current state of the system and adds noise as specified by the coprresponding
 * reinforcement learning system.
 * @details  Depending on the reinforcement learning algorithm used, the prediction is performed by the main network
 * (not the target network). In contrast to \p torchfort_rl_off_policy_predict, this routine adds noise and thus is
 * called explorative. The kind of noise is specified during system creation.
 *
 * @param[in] name The name of system instance to use, as defined during system creation.
 * @param[in] state A pointer to a memory buffer containing state data.
 * @param[in] state_dim Rank of the state data.
 * @param[in] state_shape A pointer to an array specifying the shape of the state data. Length should be equal to the
 * rank of the state data.
 * @param[in,out] action A pointer to a memory buffer to write action data.
 * @param[in] action_dim Rank of the action data.
 * @param[in] action_shape A pointer to an array specifying the shape of the action data. Length should be equal to the
 * rank of the action data.
 * @param[out] dtype The TorchFort datatype to use for this operation.
 * @param[out] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_predict_explore(const char* name, void* state, size_t state_dim,
                                                           int64_t* state_shape, void* action, size_t action_dim,
                                                           int64_t* action_shape, torchfort_datatype_t dtype,
                                                           cudaStream_t stream);

torchfort_result_t torchfort_rl_off_policy_predict_explore_F(const char* name, void* state, size_t state_dim,
                                                             int64_t* state_shape, void* action, size_t action_dim,
                                                             int64_t* action_shape, torchfort_datatype_t dtype,
                                                             cudaStream_t stream);

/**
 * @brief Suggests an action based on the current state of the system.
 * @details Depending on the algorithm used, the prediction is performed by the target network.
 * In contrast to \p torchfort_rl_off_policy_predict_explore, this routine does not add noise, which means it is
 * exploitative.
 *
 * @param[in] name The name of system instance to use, as defined during system creation.
 * @param[in] state A pointer to a memory buffer containing state data.
 * @param[in] state_dim Rank of the state data.
 * @param[in] state_shape A pointer to an array specifying the shape of the state data. Length should be equal to the
 * rank of the state data.
 * @param[in,out] action A pointer to a memory buffer to write action data.
 * @param[in] action_dim Rank of the action data.
 * @param[in] action_shape A pointer to an array specifying the shape of the action data. Length should be equal to the
 * rank of the action data.
 * @param[out] dtype The TorchFort datatype to use for this operation.
 * @param[out] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_predict(const char* name, void* state, size_t state_dim,
                                                   int64_t* state_shape, void* action, size_t action_dim,
                                                   int64_t* action_shape, torchfort_datatype_t dtype,
                                                   cudaStream_t stream);

torchfort_result_t torchfort_rl_off_policy_predict_F(const char* name, void* state, size_t state_dim,
                                                     int64_t* state_shape, void* action, size_t action_dim,
                                                     int64_t* action_shape, torchfort_datatype_t dtype,
                                                     cudaStream_t stream);

/**
 * @brief Predicts the future reward based on the current state and selected action
 * @details Depending on the learning algorithm, the routine queries the target critic networks for this.
 * The routine averages the predictions over all critics.
 *
 * @param[in] name The name of system instance to use, as defined during system creation.
 * @param[in] state A pointer to a memory buffer containing state data.
 * @param[in] state_dim Rank of the state data.
 * @param[in] state_shape A pointer to an array specifying the shape of the state data. Length should be equal to the
 * rank of the state data.
 * @param[in] action A pointer to a memory buffer containing action data.
 * @param[in] action_dim Rank of the action data.
 * @param[in] action_shape A pointer to an array specifying the shape of the action data. Length should be equal to the
 * rank of the action data.
 * @param[in, out] reward A pointer to a memory buffer to write reward data.
 * @param[in] reward_dim Rank of the reward data.
 * @param[in] reward_shape A pointer to an array specifying the shape of the reward data. Length should be equal to the
 * rank of the reward data.
 * @param[out] dtype The TorchFort datatype to use for this operation.
 * @param[out] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_evaluate(const char* name, void* state, size_t state_dim,
                                                    int64_t* state_shape, void* action, size_t action_dim,
                                                    int64_t* action_shape, void* reward, size_t reward_dim,
                                                    int64_t* reward_shape, torchfort_datatype_t dtype,
                                                    cudaStream_t stream);

torchfort_result_t torchfort_rl_off_policy_evaluate_F(const char* name, void* state, size_t state_dim,
                                                      int64_t* state_shape, void* action, size_t action_dim,
                                                      int64_t* action_shape, void* reward, size_t reward_dim,
                                                      int64_t* reward_shape, torchfort_datatype_t dtype,
                                                      cudaStream_t stream);

// RL replay buffer update functions
/**
 * @brief Adds a new \f$(s, a, s', r, d)\f$ tuple to the replay buffer
 * @details Here \f$s\f$ (\p state_old) is the state for which action \f$a\f$ (\p action_old) was taken, leading to
 * \f$s'\f$ (\p state_new) and receiving reward \f$r\f$ (\p reward). The terminal state flag \f$d\f$ (\p final_state)
 * specifies whether \f$s'\f$ is the final state in the episode.
 *
 * @param[in] name The name of system instance to use, as defined during system creation.
 * @param[in] state_old A pointer to a memory buffer containing previous state data.
 * @param[in] state_new A pointer to a memory buffer containing new state data.
 * @param[in] state_dim Rank of the state data.
 * @param[in] state_shape A pointer to an array specifying the shape of the state data. Length should be equal to the
 * rank of the \p state_old and \p state_new data.
 * @param[in] action_old A pointer to a memory buffer containing action data.
 * @param[in] action_dim Rank of the action data.
 * @param[in] action_shape A pointer to an array specifying the shape of the action data. Length should be equal to the
 * rank of the action data.
 * @param[in] reward A pointer to a memory buffer with reward data.
 * @param[in] final_state A flag indicating whether \pstate_new is the final state in the current episode (set to \p
 * true if it is the final state, otherwise \p false).
 * @param[out] dtype The TorchFort datatype to use for this operation.
 * @param[out] stream CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_update_replay_buffer(const char* name, void* state_old, void* state_new,
                                                                size_t state_dim, int64_t* state_shape,
                                                                void* action_old, size_t action_dim,
                                                                int64_t* action_shape, const void* reward,
                                                                bool final_state, torchfort_datatype_t dtype,
                                                                cudaStream_t stream);

torchfort_result_t torchfort_rl_off_policy_update_replay_buffer_F(const char* name, void* state_old, void* state_new,
                                                                  size_t state_dim, int64_t* state_shape,
                                                                  void* action_old, size_t action_dim,
                                                                  int64_t* action_shape, const void* reward,
                                                                  bool final_state, torchfort_datatype_t dtype,
                                                                  cudaStream_t stream);

// RL Checkpoint save and loading functions
/**
 * @brief Saves a reinforcement learning training checkpoint to a directory.
 * @details This method saves all models (policies, critics, target models if available) together with their
 * corresponding optimizer and LR scheduler states. It also saves the state of the replay buffer, to allow for smooth
 * restarts of reinforcement learning training processes. This function should be used in conjunction with \p
 * torchfort_rl_off_policy_load_checkpoint.
 *
 * @param[in] name The name of a system instance to save, as defined during system creation.
 * @param[in] checkpoint_dir A filesystem path to a directory to save the checkpoint data to.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_save_checkpoint(const char* name, const char* checkpoint_dir);

/**
 * @brief Restores a reinforcement learning system from a checkpoint.
 * @details This method restores all models (policies, critics, target models if available) together with their
 * corresponding optimizer and LR scheduler states. It also fully restores the state of the replay buffer, but not the
 * current RNG seed. This function should be used in conjunction with \p torchfort_rl_off_policy_save_checkpoint.
 *
 * @param[in] name The name of a system instance to restore the data for, as defined during system creation.
 * @param[in] checkpoint_dir A filesystem path to a directory which contains the checkpoint data to load.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_load_checkpoint(const char* name, const char* checkpoint_dir);

// RL miscellaneous utility functions
/**
 * @brief Queries a reinforcement learning system for rediness to start training
 * @details A user should call this method before starting training to make sure the reinforcement learning system is
 * ready. This method ensures that the replay buffer is filled sufficiently with exploration data as specified during
 * system creation.
 *
 * @param[in] name The name of a system instance to restore the data for, as defined during system creation
 * @param[out] ready A flag indicating whether the system is ready to train (\p true means it is ready to train)
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
torchfort_result_t torchfort_rl_off_policy_is_ready(const char* name, bool& ready);

// RL Weights and Bias Logging functions
/**
 * @brief Write an integer value to a Weights and Bias log using the system logging tag.  \p *_float and \p *_double
 * variants to write \p float and \p double values respectively.
 *
 * @param[in] name The name of system instance to associate this metric value with, as defined during system creation.
 * @param[in] metric_name Metric label.
 * @param[in] step Training/inference step to associate with metric value.
 * @param[in] value Metric value to log.
 *
 * @return \p TORCHFORT_RESULT_SUCCESS on success or error code on failure.
 */
RL_WANDB_LOG_PROTO(int)
RL_WANDB_LOG_PROTO(float)
RL_WANDB_LOG_PROTO(double)

#ifdef __cplusplus
}
#endif
