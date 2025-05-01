! SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

module torchfort
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
#ifdef _CUDA
  use cudafor
#endif
  implicit none

  ! enum for torchfort supported data types
  enum, bind(c) ! torchfort_data_type
    enumerator :: TORCHFORT_FLOAT = -1
    enumerator :: TORCHFORT_DOUBLE = -2
    enumerator :: TORCHFORT_INT32 = -3
    enumerator :: TORCHFORT_INT64 = -4
  end enum

  ! enum for torchfort supported device types
  enum, bind(c) ! torchfort_device
    enumerator :: TORCHFORT_DEVICE_CPU = -1
  end enum

  ! enum for torchfort return values
  enum, bind(c) ! torchfort_result
    enumerator :: TORCHFORT_RESULT_SUCCESS = 0
    enumerator :: TORCHFORT_RESULT_INVALID_USAGE = 1
    enumerator :: TORCHFORT_RESULT_NOT_SUPPORTED = 2
    enumerator :: TORCHFORT_RESULT_INTERNAL_ERROR = 3
    enumerator :: TORCHFORT_RESULT_CUDA_ERROR = 4
    enumerator :: TORCHFORT_RESULT_MPI_ERROR = 5
    enumerator :: TORCHFORT_RESULT_NCCL_ERROR = 6
  end enum

  ! MPI-related types
#ifndef MPICH
  type, bind(c) :: MPI_C_Comm
    integer(c_int64_t) :: comm
  end type MPI_C_Comm
#else
  type, bind(c) :: MPI_C_Comm
    integer(c_int) :: comm
  end type MPI_C_Comm
#endif

  type, bind(c) :: MPI_F_Comm
    integer(c_int) :: comm
  end type MPI_F_Comm

  ! MPI_Comm conversion functions
#ifndef MPICH
  interface
    function MPI_Comm_f2c(fcomm) bind(C, name='MPI_Comm_f2c') result(res)
      import
      type(MPI_F_comm), value :: fcomm
      type(MPI_C_comm) :: res
    end function MPI_Comm_f2c

  end interface
#endif

  ! Opaque handle to torchfort_tensor_list
  type, bind(c) :: torchfort_tensor_list
    type(c_ptr) :: member
  end type torchfort_tensor_list

  ! Interfaces to C functions
  interface
    function torchfort_wandb_log_int_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_wandb_log_int")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      integer(c_int32_t), value :: val
      integer(c_int) :: res
    end function torchfort_wandb_log_int_c

    function torchfort_wandb_log_float_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_wandb_log_float")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      real(c_float), value :: val
      integer(c_int) :: res
    end function torchfort_wandb_log_float_c

    function torchfort_wandb_log_double_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_wandb_log_double")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      real(c_double), value :: val
      integer(c_int) :: res
    end function torchfort_wandb_log_double_c

    function torchfort_inference_c(mname, input, input_dim, input_shape, &
                                   output, output_dim, output_shape, dtype, stream) result(res) &
      bind(C, name="torchfort_inference_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)input, (dk)output
      !GCC$ attributes no_arg_check :: input, output
      real(c_float) :: input(*), output(*)
      integer(c_size_t), value :: input_dim, output_dim
      integer(c_int64_t) :: input_shape(*), output_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_inference_c

    function torchfort_inference_multiarg_c(mname, inputs, outputs, stream) result(res) &
      bind(C, name="torchfort_inference_multiarg")
      import
      type(*) :: mname(*)
      type(torchfort_tensor_list), value :: inputs, outputs
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_inference_multiarg_c

    function torchfort_train_c(mname, input, input_dim, input_shape, &
                               label, label_dim, label_shape, &
                               loss_val, dtype, stream) result(res) &
      bind(C, name="torchfort_train_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)input, (dk)label, (k)loss_val
      !GCC$ attributes no_arg_check :: input, label, loss_val
      real(c_float) :: input(*), label(*)
      real(c_float) :: loss_val
      integer(c_size_t), value :: input_dim, label_dim
      integer(c_int64_t) :: input_shape(*), label_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_train_c

    function torchfort_train_multiarg_c(mname, inputs, labels, &
                                        loss_val, extra_loss_args, stream) result(res) &
      bind(C, name="torchfort_train_multiarg")
      import
      type(*) :: mname(*)
      type(torchfort_tensor_list), value :: inputs, labels
      real(c_float) :: loss_val
      type(torchfort_tensor_list), value :: extra_loss_args
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_train_multiarg_c

    function torchfort_set_cudnn_benchmark_c(flag) result(res) &
      bind(C, name="torchfort_set_cudnn_benchmark")
      import
      logical :: flag
      integer(c_int) :: res
    end function torchfort_set_cudnn_benchmark_c

    function torchfort_set_cuda_allow_tf32_c(flag) result(res) &
      bind(C, name="torchfort_set_cuda_allow_tf32")
      import
      logical :: flag
      integer(c_int) :: res
    end function torchfort_set_cuda_allow_tf32_c

    function torchfort_set_manual_seed_c(seed) result(res) &
      bind(C, name="torchfort_set_manual_seed")
      import
      integer(c_int) :: seed
      integer(c_int) :: res
    end function torchfort_set_manual_seed_c

    function torchfort_set_cuda_manual_seed_c(seed) result(res) &
      bind(C, name="torchfort_set_cuda_manual_seed")
      import
      integer(c_int) :: seed
      integer(c_int) :: res
    end function torchfort_set_cuda_manual_seed_c

    function torchfort_create_model_c(mname, fname, dev) result(res) &
      bind(C, name="torchfort_create_model")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      integer(c_int), value :: dev
      integer(c_int) :: res
    end function torchfort_create_model_c

    function torchfort_create_distributed_model_c(mname, fname, mpi_comm, dev) result(res) &
      bind(C, name="torchfort_create_distributed_model")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      integer(c_int), value :: dev
      type(MPI_C_Comm), value :: mpi_comm
      integer(c_int) :: res
    end function torchfort_create_distributed_model_c

    function torchfort_save_model_c(mname, fname) result(res) &
      bind(C, name="torchfort_save_model")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      integer(c_int) :: res
    end function torchfort_save_model_c

    function torchfort_load_model_c(mname, fname) result(res) &
      bind(C, name="torchfort_load_model")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      integer(c_int) :: res
    end function torchfort_load_model_c

    function torchfort_save_checkpoint_c(mname, checkpoint_dir) result(res) &
      bind(C, name="torchfort_save_checkpoint")
      import
      type(*) :: mname(*)
      type(*) :: checkpoint_dir(*)
      integer(c_int) :: res
    end function torchfort_save_checkpoint_c

    function torchfort_load_checkpoint_c(mname, checkpoint_dir, &
                                         step_train, step_inference) result(res) &
      bind(C, name="torchfort_load_checkpoint")
      import
      type(*) :: mname(*)
      type(*) :: checkpoint_dir(*)
      integer(c_int64_t)  :: step_train
      integer(c_int64_t)  :: step_inference
      integer(c_int) :: res
    end function torchfort_load_checkpoint_c

    ! RL off-policy
    ! logging
    function torchfort_rl_off_policy_wandb_log_int_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_rl_off_policy_wandb_log_int")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      integer(c_int32_t), value :: val
      integer(c_int) :: res
    end function torchfort_rl_off_policy_wandb_log_int_c

    function torchfort_rl_off_policy_wandb_log_float_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_rl_off_policy_wandb_log_float")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      real(c_float), value :: val
      integer(c_int) :: res
    end function torchfort_rl_off_policy_wandb_log_float_c

    function torchfort_rl_off_policy_wandb_log_double_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_rl_off_policy_wandb_log_double")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      real(c_double), value :: val
      integer(c_int) :: res
    end function torchfort_rl_off_policy_wandb_log_double_c

    ! creation
    function torchfort_rl_off_policy_create_system_c(mname, fname, model_dev, rb_dev) result(res) &
      bind(C, name="torchfort_rl_off_policy_create_system")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      integer(c_int), value :: model_dev, rb_dev
      integer(c_int) :: res
    end function torchfort_rl_off_policy_create_system_c

    function torchfort_rl_off_policy_create_distributed_system_c(mname, fname, mpi_comm, model_dev, rb_dev) result(res) &
      bind(C, name="torchfort_rl_off_policy_create_distributed_system")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      type(MPI_C_Comm), value :: mpi_comm
      integer(c_int), value :: model_dev, rb_dev
      integer(c_int) :: res
    end function torchfort_rl_off_policy_create_distributed_system_c

    ! saving and loading
    function torchfort_rl_off_policy_save_checkpoint_c(mname, checkpoint_dir) result(res) &
      bind(C, name="torchfort_rl_off_policy_save_checkpoint")
      import
      type(*) :: mname(*)
      type(*) :: checkpoint_dir(*)
      integer(c_int) :: res
    end function torchfort_rl_off_policy_save_checkpoint_c

    function torchfort_rl_off_policy_load_checkpoint_c(mname, checkpoint_dir) result(res) &
      bind(C, name="torchfort_rl_off_policy_load_checkpoint")
      import
      type(*) :: mname(*)
      type(*) :: checkpoint_dir(*)
      integer(c_int) :: res
    end function torchfort_rl_off_policy_load_checkpoint_c

    ! training
    function torchfort_rl_off_policy_update_replay_buffer_c(mname, &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, dtype, stream) result(res) &
      bind(C, name="torchfort_rl_off_policy_update_replay_buffer_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state_old, (dk)state_new, (dk)act_old, (k)reward
      !GCC$ attributes no_arg_check :: state_old, state_new, act_old, reward
      real(c_float) :: state_old(*), state_new(*), act_old(*)
      real(c_float) :: reward
      logical(c_bool), value :: cterminal
      integer(c_size_t), value :: state_dim, act_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_off_policy_update_replay_buffer_c

    function torchfort_rl_off_policy_update_replay_buffer_multi_c(mname, &
                                                                  state_old, state_new, state_dim, state_shape, &
                                                                  act_old, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  cterminal, cterminal_dim, cterminal_shape, &
                                                                  dtype, stream) result(res) &
      bind(C, name="torchfort_rl_off_policy_update_replay_buffer_multi_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state_old, (dk)state_new, (dk)act_old, (dk)reward, (dk)cterminal
      !GCC$ attributes no_arg_check :: state_old, state_new, act_old, reward, cterminal
      real(c_float) :: state_old(*), state_new(*), act_old(*), reward(*), cterminal(*)
      integer(c_size_t), value :: state_dim, act_dim, reward_dim, cterminal_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*), reward_shape(*), cterminal_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_off_policy_update_replay_buffer_multi_c

    function torchfort_rl_off_policy_is_ready_c(mname, ready) result(res) &
      bind(C, name="torchfort_rl_off_policy_is_ready")
      import
      type(*) :: mname(*)
      logical :: ready
      integer(c_int) :: res
    end function torchfort_rl_off_policy_is_ready_c

    function torchfort_rl_off_policy_train_step_float_c(mname, p_loss_val, q_loss_val, stream) result(res) &
      bind(C, name="torchfort_rl_off_policy_train_step")
      import
      type(*) :: mname(*)
      real(c_float) :: p_loss_val, q_loss_val
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_off_policy_train_step_float_c

    ! prediction
    function torchfort_rl_off_policy_predict_explore_c(mname, state, state_dim, state_shape, &
                                                       act, act_dim, act_shape, dtype, stream) result(res) &
      bind(C, name="torchfort_rl_off_policy_predict_explore_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act
      !GCC$ attributes no_arg_check :: state, act
      real(c_float) :: state(*), act(*)
      integer(c_size_t), value :: state_dim, act_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_off_policy_predict_explore_c

    function torchfort_rl_off_policy_predict_c(mname, state, state_dim, state_shape, &
                                               act, act_dim, act_shape, dtype, stream) result(res) &
      bind(C, name="torchfort_rl_off_policy_predict_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act
      !GCC$ attributes no_arg_check :: state, act
      real(c_float) :: state(*), act(*)
      integer(c_size_t), value :: state_dim, act_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_off_policy_predict_c

    function torchfort_rl_off_policy_evaluate_c(mname, state, state_dim, state_shape, &
                                                act, act_dim, act_shape, &
                                                reward, reward_dim, reward_shape, &
                                                dtype, stream) result(res) &
      bind(C, name="torchfort_rl_off_policy_evaluate_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr state, act, reward
      !GCC$ attributes no_arg_check :: state, act, reward
      real(c_float) :: state(*), act(*), reward(*)
      integer(c_size_t), value :: state_dim, act_dim, reward_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*), reward_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_off_policy_evaluate_c

    ! RL on-policy
    ! logging
    function torchfort_rl_on_policy_wandb_log_int_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_rl_on_policy_wandb_log_int")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      integer(c_int32_t), value :: val
      integer(c_int) :: res
    end function torchfort_rl_on_policy_wandb_log_int_c

    function torchfort_rl_on_policy_wandb_log_float_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_rl_on_policy_wandb_log_float")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      real(c_float), value :: val
      integer(c_int) :: res
    end function torchfort_rl_on_policy_wandb_log_float_c

    function torchfort_rl_on_policy_wandb_log_double_c(mname, metric_name, step, val) result(res) &
      bind(C, name="torchfort_rl_on_policy_wandb_log_double")
      import
      type(*) :: mname(*), metric_name(*)
      integer(c_int64_t), value :: step
      real(c_double), value :: val
      integer(c_int) :: res
    end function torchfort_rl_on_policy_wandb_log_double_c

    ! creation
    function torchfort_rl_on_policy_create_system_c(mname, fname, model_dev, rb_dev) result(res) &
      bind(C, name="torchfort_rl_on_policy_create_system")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      integer(c_int), value :: model_dev, rb_dev
      integer(c_int) :: res
    end function torchfort_rl_on_policy_create_system_c

    function torchfort_rl_on_policy_create_distributed_system_c(mname, fname, mpi_comm, model_dev, rb_dev) result(res) &
      bind(C, name="torchfort_rl_on_policy_create_distributed_system")
      import
      type(*) :: mname(*)
      type(*) :: fname(*)
      type(MPI_C_Comm), value :: mpi_comm
      integer(c_int), value :: model_dev, rb_dev
      integer(c_int) :: res
    end function torchfort_rl_on_policy_create_distributed_system_c

    ! saving and loading
    function torchfort_rl_on_policy_save_checkpoint_c(mname, checkpoint_dir) result(res) &
      bind(C, name="torchfort_rl_on_policy_save_checkpoint")
      import
      type(*) :: mname(*)
      type(*) :: checkpoint_dir(*)
      integer(c_int) :: res
    end function torchfort_rl_on_policy_save_checkpoint_c

    function torchfort_rl_on_policy_load_checkpoint_c(mname, checkpoint_dir) result(res) &
      bind(C, name="torchfort_rl_on_policy_load_checkpoint")
      import
      type(*) :: mname(*)
      type(*) :: checkpoint_dir(*)
      integer(c_int) :: res
    end function torchfort_rl_on_policy_load_checkpoint_c

    ! training
    function torchfort_rl_on_policy_update_rollout_buffer_c(mname, &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, dtype, stream) result(res) &
      bind(C, name="torchfort_rl_on_policy_update_rollout_buffer_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act, (k)reward
      !GCC$ attributes no_arg_check :: state, act, reward
      real(c_float) :: state(*), act(*)
      real(c_float) :: reward
      logical(c_bool), value :: cterminal
      integer(c_size_t), value :: state_dim, act_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_on_policy_update_rollout_buffer_c

    function torchfort_rl_on_policy_update_rollout_buffer_multi_c(mname, &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  cterminal, cterminal_dim, cterminal_shape, &
                                                                  dtype, stream) result(res) &
      bind(C, name="torchfort_rl_on_policy_update_rollout_buffer_multi_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act, (dk)reward, (dk)cterminal
      !GCC$ attributes no_arg_check :: state, act, reward, cterminal
      real(c_float) :: state(*), act(*), reward(*), cterminal(*)
      integer(c_size_t), value :: state_dim, act_dim, reward_dim, cterminal_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*), reward_shape(*), cterminal_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_on_policy_update_rollout_buffer_multi_c

    function torchfort_rl_on_policy_reset_rollout_buffer_c(mname) result(res) &
      bind(C, name="torchfort_rl_on_policy_reset_rollout_buffer")
      import
      type(*) :: mname(*)
      integer(c_int) :: res
    end function torchfort_rl_on_policy_reset_rollout_buffer_c

    function torchfort_rl_on_policy_is_ready_c(mname, ready) result(res) &
      bind(C, name="torchfort_rl_on_policy_is_ready")
      import
      type(*) :: mname(*)
      logical :: ready
      integer(c_int) :: res
    end function torchfort_rl_on_policy_is_ready_c

    function torchfort_rl_on_policy_train_step_float_c(mname, p_loss_val, q_loss_val, stream) result(res) &
      bind(C, name="torchfort_rl_on_policy_train_step")
      import
      type(*) :: mname(*)
      real(c_float) :: p_loss_val, q_loss_val
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_on_policy_train_step_float_c

    ! prediction
    function torchfort_rl_on_policy_predict_explore_c(mname, state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, dtype, stream) result(res) &
      bind(C, name="torchfort_rl_on_policy_predict_explore_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act
      !GCC$ attributes no_arg_check :: state, act
      real(c_float) :: state(*), act(*)
      integer(c_size_t), value :: state_dim, act_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_on_policy_predict_explore_c

    function torchfort_rl_on_policy_predict_c(mname, state, state_dim, state_shape, &
                                               act, act_dim, act_shape, dtype, stream) result(res) &
      bind(C, name="torchfort_rl_on_policy_predict_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act
      !GCC$ attributes no_arg_check :: state, act
      real(c_float) :: state(*), act(*)
      integer(c_size_t), value :: state_dim, act_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_on_policy_predict_c

    function torchfort_rl_on_policy_evaluate_c(mname, state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               dtype, stream) result(res) &
      bind(C, name="torchfort_rl_on_policy_evaluate_F")
      import
      type(*) :: mname(*)
      !dir$ ignore_tkr (dk)state, (dk)act, (dk)reward
      !GCC$ attributes no_arg_check :: state, act, reward
      real(c_float) :: state(*), act(*), reward(*)
      integer(c_size_t), value :: state_dim, act_dim, reward_dim
      integer(c_int64_t) :: state_shape(*), act_shape(*), reward_shape(*)
      integer(c_int), value :: dtype
      integer(int64), value :: stream
      integer(c_int) :: res
    end function torchfort_rl_on_policy_evaluate_c

    ! Tensor list management
    function torchfort_tensor_list_create(tensor_list) result(res) &
      bind(C, name="torchfort_tensor_list_create")
      import
      type(torchfort_tensor_list) :: tensor_list
      integer(c_int) :: res
    end function torchfort_tensor_list_create

    function torchfort_tensor_list_destroy(tensor_list) result(res) &
      bind(C, name="torchfort_tensor_list_destroy")
      import
      type(torchfort_tensor_list), value :: tensor_list
      integer(c_int) :: res
    end function torchfort_tensor_list_destroy

    function torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                                                dtype) result(res) &
      bind(C, name="torchfort_tensor_list_add_tensor_F")
      import
      type(torchfort_tensor_list), value :: tensor_list
      !dir$ ignore_tkr (tkd)data_arr
      !GCC$ attributes no_arg_check :: data_arr
      real(c_float) :: data_arr(*)
      integer(c_size_t), value :: data_dim
      integer(c_int64_t) :: data_shape(*)
      integer(c_int), value :: dtype
      integer(c_int) :: res
    end function torchfort_tensor_list_add_tensor_c

  end interface

  ! Generic interface for W&B logging
  interface torchfort_wandb_log
    module procedure torchfort_wandb_log_int
    module procedure torchfort_wandb_log_float
    module procedure torchfort_wandb_log_double
    module procedure torchfort_wandb_log_float_int32step
    module procedure torchfort_wandb_log_double_int32step
  end interface

  ! Generic interface for inference
  interface torchfort_inference
    module procedure torchfort_inference_float_2d
    module procedure torchfort_inference_double_2d
    module procedure torchfort_inference_float_3d
    module procedure torchfort_inference_double_3d
    module procedure torchfort_inference_float_4d
    module procedure torchfort_inference_double_4d
    module procedure torchfort_inference_float_5d
    module procedure torchfort_inference_double_5d
#ifdef _CUDA
    module procedure torchfort_inference_float_2d_dev
    module procedure torchfort_inference_double_2d_dev
    module procedure torchfort_inference_float_3d_dev
    module procedure torchfort_inference_double_3d_dev
    module procedure torchfort_inference_float_4d_dev
    module procedure torchfort_inference_double_4d_dev
    module procedure torchfort_inference_float_5d_dev
    module procedure torchfort_inference_double_5d_dev
#endif
  end interface torchfort_inference

  ! Generic interface for training
  interface torchfort_train
    module procedure torchfort_train_float_2d
    module procedure torchfort_train_double_2d
    module procedure torchfort_train_float_3d
    module procedure torchfort_train_double_3d
    module procedure torchfort_train_float_4d
    module procedure torchfort_train_double_4d
    module procedure torchfort_train_float_5d
    module procedure torchfort_train_double_5d
#ifdef _CUDA
    module procedure torchfort_train_float_2d_dev
    module procedure torchfort_train_double_2d_dev
    module procedure torchfort_train_float_3d_dev
    module procedure torchfort_train_double_3d_dev
    module procedure torchfort_train_float_4d_dev
    module procedure torchfort_train_double_4d_dev
    module procedure torchfort_train_float_5d_dev
    module procedure torchfort_train_double_5d_dev
#endif
  end interface torchfort_train

  ! Generic interface for distributed setup
  interface torchfort_create_distributed_model
    module procedure torchfort_create_distributed_model_MPI_F
    module procedure torchfort_create_distributed_model_MPI_F08
    module procedure torchfort_create_distributed_model_type
  end interface torchfort_create_distributed_model

  ! Generic interface for load checkpoint
  interface torchfort_load_checkpoint
    module procedure torchfort_load_checkpoint_int64step
    module procedure torchfort_load_checkpoint_int32step
  end interface torchfort_load_checkpoint

  ! interfaces for RL off-policy routines
  ! creation
  ! Generic interface for distributed setup
  interface torchfort_rl_off_policy_create_distributed_system
    module procedure torchfort_rl_off_policy_create_distributed_system_MPI_F
    module procedure torchfort_rl_off_policy_create_distributed_system_MPI_F08
    module procedure torchfort_rl_off_policy_create_distributed_system_type
  end interface torchfort_rl_off_policy_create_distributed_system

  interface torchfort_rl_off_policy_wandb_log
    module procedure torchfort_rl_off_policy_wandb_log_int
    module procedure torchfort_rl_off_policy_wandb_log_float
    module procedure torchfort_rl_off_policy_wandb_log_double
    module procedure torchfort_rl_off_policy_wandb_log_float_int32step
    module procedure torchfort_rl_off_policy_wandb_log_double_int32step
  end interface torchfort_rl_off_policy_wandb_log

  ! Generic interface for training
  interface torchfort_rl_off_policy_update_replay_buffer
     ! single env
     module procedure torchfort_rl_off_policy_update_replay_buffer_float_1d_1d
     module procedure torchfort_rl_off_policy_update_replay_buffer_float_3d_1d
     module procedure torchfort_rl_off_policy_update_replay_buffer_float_3d_3d
#ifdef _CUDA
     module procedure torchfort_rl_off_policy_update_replay_buffer_float_1d_1d_dev
     module procedure torchfort_rl_off_policy_update_replay_buffer_float_3d_1d_dev
     module procedure torchfort_rl_off_policy_update_replay_buffer_float_3d_3d_dev
#endif
     ! multi env
     module procedure torchfort_rl_off_policy_update_replay_buffer_multi_float_1d_1d
     module procedure torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_1d
     module procedure torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_3d
#ifdef _CUDA
     module procedure torchfort_rl_off_policy_update_replay_buffer_multi_float_1d_1d_dev
     module procedure torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_1d_dev
     module procedure torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_3d_dev
#endif
  end interface torchfort_rl_off_policy_update_replay_buffer

  interface torchfort_rl_off_policy_train_step
     module procedure torchfort_rl_off_policy_train_step_float
  end interface torchfort_rl_off_policy_train_step

  interface torchfort_rl_off_policy_predict_explore
     module procedure torchfort_rl_off_policy_predict_explore_float_2d_2d
     module procedure torchfort_rl_off_policy_predict_explore_float_4d_4d
     module procedure torchfort_rl_off_policy_predict_explore_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_off_policy_predict_explore_float_2d_2d_dev
     module procedure torchfort_rl_off_policy_predict_explore_float_4d_4d_dev
     module procedure torchfort_rl_off_policy_predict_explore_float_4d_2d_dev
#endif
  end interface torchfort_rl_off_policy_predict_explore

  interface torchfort_rl_off_policy_predict
     module procedure torchfort_rl_off_policy_predict_float_4d_4d
     module procedure torchfort_rl_off_policy_predict_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_off_policy_predict_float_4d_4d_dev
     module procedure torchfort_rl_off_policy_predict_float_4d_2d_dev
#endif
  end interface torchfort_rl_off_policy_predict

  interface torchfort_rl_off_policy_evaluate
     module procedure torchfort_rl_off_policy_evaluate_float_2d_2d
     module procedure torchfort_rl_off_policy_evaluate_float_4d_4d
     module procedure torchfort_rl_off_policy_evaluate_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_off_policy_evaluate_float_2d_2d_dev
     module procedure torchfort_rl_off_policy_evaluate_float_4d_4d_dev
     module procedure torchfort_rl_off_policy_evaluate_float_4d_2d_dev
#endif
  end interface torchfort_rl_off_policy_evaluate

  ! interfaces for RL on-policy routines
  ! creation
  ! Generic interface for distributed setup
  interface torchfort_rl_on_policy_create_distributed_system
    module procedure torchfort_rl_on_policy_create_distributed_system_MPI_F
    module procedure torchfort_rl_on_policy_create_distributed_system_MPI_F08
    module procedure torchfort_rl_on_policy_create_distributed_system_type
  end interface torchfort_rl_on_policy_create_distributed_system

  interface torchfort_rl_on_policy_wandb_log
    module procedure torchfort_rl_on_policy_wandb_log_int
    module procedure torchfort_rl_on_policy_wandb_log_float
    module procedure torchfort_rl_on_policy_wandb_log_double
    module procedure torchfort_rl_on_policy_wandb_log_float_int32step
    module procedure torchfort_rl_on_policy_wandb_log_double_int32step
  end interface torchfort_rl_on_policy_wandb_log

  ! Generic interface for training
  interface torchfort_rl_on_policy_update_rollout_buffer
     ! single env
     module procedure torchfort_rl_on_policy_update_rollout_buffer_float_1d_1d
     module procedure torchfort_rl_on_policy_update_rollout_buffer_float_3d_1d
     module procedure torchfort_rl_on_policy_update_rollout_buffer_float_3d_3d
#ifdef _CUDA
     module procedure torchfort_rl_on_policy_update_rollout_buffer_float_1d_1d_dev
     module procedure torchfort_rl_on_policy_update_rollout_buffer_float_3d_1d_dev
     module procedure torchfort_rl_on_policy_update_rollout_buffer_float_3d_3d_dev
#endif
     ! multi env
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_1d_1d
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_1d
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_3d
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_1d_1d_dev
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_1d_dev
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_3d_dev
     module procedure torchfort_rl_on_policy_update_rollout_buffer_multi_float_4d_2d_dev
#endif
  end interface torchfort_rl_on_policy_update_rollout_buffer

  interface torchfort_rl_on_policy_train_step
     module procedure torchfort_rl_on_policy_train_step_float
  end interface torchfort_rl_on_policy_train_step

  interface  torchfort_rl_on_policy_predict_explore
     module procedure torchfort_rl_on_policy_predict_explore_float_2d_2d
     module procedure torchfort_rl_on_policy_predict_explore_float_4d_4d
     module procedure torchfort_rl_on_policy_predict_explore_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_on_policy_predict_explore_float_4d_4d_dev
     module procedure torchfort_rl_on_policy_predict_explore_float_4d_4d_dev
     module procedure torchfort_rl_on_policy_predict_explore_float_4d_2d_dev
#endif
  end interface torchfort_rl_on_policy_predict_explore

  interface  torchfort_rl_on_policy_predict
     module procedure torchfort_rl_on_policy_predict_float_2d_2d
     module procedure torchfort_rl_on_policy_predict_float_4d_4d
     module procedure torchfort_rl_on_policy_predict_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_on_policy_predict_float_2d_2d_dev
     module procedure torchfort_rl_on_policy_predict_float_4d_4d_dev
     module procedure torchfort_rl_on_policy_predict_float_4d_2d_dev
#endif
  end interface torchfort_rl_on_policy_predict

  interface  torchfort_rl_on_policy_evaluate
     module procedure torchfort_rl_on_policy_evaluate_float_2d_2d
     module procedure torchfort_rl_on_policy_evaluate_float_4d_4d
     module procedure torchfort_rl_on_policy_evaluate_float_4d_2d
#ifdef _CUDA
     module procedure torchfort_rl_on_policy_evaluate_float_2d_2d_dev
     module procedure torchfort_rl_on_policy_evaluate_float_4d_4d_dev
     module procedure torchfort_rl_on_policy_evaluate_float_4d_2d_dev
#endif
  end interface torchfort_rl_on_policy_evaluate

  ! Generic interface for tensor list management
  interface torchfort_tensor_list_add_tensor
    module procedure torchfort_tensor_list_add_tensor_float_1d
    module procedure torchfort_tensor_list_add_tensor_double_1d
    module procedure torchfort_tensor_list_add_tensor_int32_1d
    module procedure torchfort_tensor_list_add_tensor_int64_1d
    module procedure torchfort_tensor_list_add_tensor_float_2d
    module procedure torchfort_tensor_list_add_tensor_double_2d
    module procedure torchfort_tensor_list_add_tensor_int32_2d
    module procedure torchfort_tensor_list_add_tensor_int64_2d
    module procedure torchfort_tensor_list_add_tensor_float_3d
    module procedure torchfort_tensor_list_add_tensor_double_3d
    module procedure torchfort_tensor_list_add_tensor_int32_3d
    module procedure torchfort_tensor_list_add_tensor_int64_3d
    module procedure torchfort_tensor_list_add_tensor_float_4d
    module procedure torchfort_tensor_list_add_tensor_double_4d
    module procedure torchfort_tensor_list_add_tensor_int32_4d
    module procedure torchfort_tensor_list_add_tensor_int64_4d
    module procedure torchfort_tensor_list_add_tensor_float_5d
    module procedure torchfort_tensor_list_add_tensor_double_5d
    module procedure torchfort_tensor_list_add_tensor_int32_5d
    module procedure torchfort_tensor_list_add_tensor_int64_5d
#ifdef _CUDA
    module procedure torchfort_tensor_list_add_tensor_float_1d_dev
    module procedure torchfort_tensor_list_add_tensor_double_1d_dev
    module procedure torchfort_tensor_list_add_tensor_int32_1d_dev
    module procedure torchfort_tensor_list_add_tensor_int64_1d_dev
    module procedure torchfort_tensor_list_add_tensor_float_2d_dev
    module procedure torchfort_tensor_list_add_tensor_double_2d_dev
    module procedure torchfort_tensor_list_add_tensor_int32_2d_dev
    module procedure torchfort_tensor_list_add_tensor_int64_2d_dev
    module procedure torchfort_tensor_list_add_tensor_float_3d_dev
    module procedure torchfort_tensor_list_add_tensor_double_3d_dev
    module procedure torchfort_tensor_list_add_tensor_int32_3d_dev
    module procedure torchfort_tensor_list_add_tensor_int64_3d_dev
    module procedure torchfort_tensor_list_add_tensor_float_4d_dev
    module procedure torchfort_tensor_list_add_tensor_double_4d_dev
    module procedure torchfort_tensor_list_add_tensor_int32_4d_dev
    module procedure torchfort_tensor_list_add_tensor_int64_4d_dev
    module procedure torchfort_tensor_list_add_tensor_float_5d_dev
    module procedure torchfort_tensor_list_add_tensor_double_5d_dev
    module procedure torchfort_tensor_list_add_tensor_int32_5d_dev
    module procedure torchfort_tensor_list_add_tensor_int64_5d_dev
#endif
  end interface torchfort_tensor_list_add_tensor

contains

  ! global routines
  function torchfort_set_cudnn_benchmark(flag) result(res)
    logical :: flag
    integer(c_int) :: res
    res = torchfort_set_cudnn_benchmark_c(flag)
  end function torchfort_set_cudnn_benchmark

  function torchfort_set_cuda_allow_tf32(flag) result(res)
    logical :: flag
    integer(c_int) :: res
    res = torchfort_set_cuda_allow_tf32_c(flag)
  end function torchfort_set_cuda_allow_tf32

  function torchfort_set_manual_seed(seed) result(res)
    integer(c_int) :: seed
    integer(c_int) :: res
    res = torchfort_set_manual_seed_c(seed)
  end function torchfort_set_manual_seed

  function torchfort_set_cuda_manual_seed(seed) result(res)
    integer(c_int) :: seed
    integer(c_int) :: res
    res = torchfort_set_cuda_manual_seed_c(seed)
  end function torchfort_set_cuda_manual_seed

  ! Setup routines
  function torchfort_create_model(mname, fname, dev) result(res)
    character(len=*) :: mname, fname
    integer(c_int) :: dev
    integer(c_int) :: res
    res = torchfort_create_model_c([trim(mname) // C_NULL_CHAR], [trim(fname) // C_NULL_CHAR], dev)
  end function torchfort_create_model

  function torchfort_create_distributed_model_MPI_F(mname, fname, comm, dev) result(res)
    character(len=*) :: mname, fname
    integer :: comm
    integer(c_int) :: dev
    integer(c_int) :: res

    type(MPI_F_Comm) :: mpi_comm_f

    mpi_comm_f%comm = comm
    res = torchfort_create_distributed_model_type(mname, fname, mpi_comm_f, dev)
  end function torchfort_create_distributed_model_MPI_F

  function torchfort_create_distributed_model_MPI_F08(mname, fname, comm, dev) result(res)
    type, bind(c) :: MPI_Comm
      integer :: MPI_VAL
    end type MPI_Comm
    character(len=*) :: mname, fname
    type(MPI_Comm) :: comm
    integer(c_int) :: dev
    integer(c_int) :: res

    type(MPI_F_Comm) :: mpi_comm_f

    mpi_comm_f%comm = comm%MPI_VAL
    res = torchfort_create_distributed_model_type(mname, fname, mpi_comm_f, dev)
  end function torchfort_create_distributed_model_MPI_F08

  function torchfort_create_distributed_model_type(mname, fname, comm, dev) result(res)
    character(len=*) :: mname, fname
    type(MPI_F_Comm) :: comm
    integer(c_int) :: dev
    integer(c_int) :: res

    type(MPI_C_Comm) :: mpi_comm_c

#ifndef MPICH
    mpi_comm_c = MPI_Comm_f2c(comm)
#else
    mpi_comm_c%comm = comm%comm
#endif
    res = torchfort_create_distributed_model_c([trim(mname) // C_NULL_CHAR], [trim(fname) // C_NULL_CHAR], &
                                               mpi_comm_c, dev)
  end function torchfort_create_distributed_model_type

  ! W&B logging routines
  function torchfort_wandb_log_int(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    integer(int32) :: val
    integer(c_int) :: res
    res = torchfort_wandb_log_int_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                    step, val)
  end function torchfort_wandb_log_int

  function torchfort_wandb_log_float(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    real(real32) :: val
    integer(c_int) :: res
    res = torchfort_wandb_log_float_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                      step, val)
  end function torchfort_wandb_log_float

  function torchfort_wandb_log_float_int32step(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int32) :: step
    real(real32) :: val
    integer(c_int) :: res
    integer(int64) :: step64
    step64 = step
    res = torchfort_wandb_log_float_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                      step64, val)
  end function torchfort_wandb_log_float_int32step

  function torchfort_wandb_log_double(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    real(real64) :: val
    integer(c_int) :: res
    res = torchfort_wandb_log_double_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                       step, val)
  end function torchfort_wandb_log_double

  function torchfort_wandb_log_double_int32step(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int32) :: step
    real(real64) :: val
    integer(c_int) :: res
    integer(int64) :: step64
    step64 = step
    res = torchfort_wandb_log_double_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                       step64, val)
  end function torchfort_wandb_log_double_int32step

  ! Inference routines
  function torchfort_inference_float_2d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :), output(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_
    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_2d

  function torchfort_inference_double_2d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :), output(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_2d

  function torchfort_inference_float_3d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :, :), output(:, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_3d

  function torchfort_inference_double_3d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :, :), output(:, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_3d

  function torchfort_inference_float_4d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :, :, :), output(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_4d

  function torchfort_inference_double_4d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :, :, :), output(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_4d

  function torchfort_inference_float_5d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :, :, :, :), output(:, :, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_5d

  function torchfort_inference_double_5d(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :, :, :, :), output(:, :, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_5d

#ifdef _CUDA
  function torchfort_inference_float_2d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :), output(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_
    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_2d_dev

  function torchfort_inference_double_2d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :), output(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_2d_dev

  function torchfort_inference_float_3d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :, :), output(:, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_3d_dev

  function torchfort_inference_double_3d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :, :), output(:, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_3d_dev

  function torchfort_inference_float_4d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :, :, :), output(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_4d_dev

  function torchfort_inference_double_4d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :, :, :), output(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_4d_dev

  function torchfort_inference_float_5d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :, :, :, :), output(:, :, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_inference_float_5d_dev

  function torchfort_inference_double_5d_dev(mname, input, output, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :, :, :, :), output(:, :, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, output_dim

    input_dim = size(shape(input))
    output_dim = size(shape(output))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
    integer(c_int64_t) :: input_shape(input_dim)
    integer(c_int64_t) :: output_shape(output_dim)

    input_shape(:) = shape(input)
    output_shape(:) = shape(output)

    res = torchfort_inference_c([trim(mname) // C_NULL_CHAR], &
                                input, input_dim, input_shape, &
                                output, output_dim, output_shape, &
                                TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_inference_double_5d_dev
#endif

  function torchfort_inference_multiarg(mname, inputs, outputs, stream) result(res)
    character(len=*) :: mname
    type(torchfort_tensor_list) :: inputs, outputs
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    stream_ = 0
    if (present(stream)) stream_ = stream

    res = torchfort_inference_multiarg_c([trim(mname) // C_NULL_CHAR], &
                                         inputs, outputs, stream_)
  end function torchfort_inference_multiarg

  ! Training routines
  function torchfort_train_float_2d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :), label(:, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res =  torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                               input, input_dim, input_shape, &
                               label, label_dim, label_shape, &
                               loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_2d

  function torchfort_train_double_2d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :), label(:, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_2d

  function torchfort_train_float_3d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :, :), label(:, :, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_3d

  function torchfort_train_double_3d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :, :), label(:, :, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_3d

  function torchfort_train_float_4d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :, :, :), label(:, :, :, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_4d

  function torchfort_train_double_4d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :, :, :), label(:, :, :, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_4d

  function torchfort_train_float_5d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32) :: input(:, :, :, :, :), label(:, :, :, :, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_5d

  function torchfort_train_double_5d(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64) :: input(:, :, :, :, :), label(:, :, :, :, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_5d

#ifdef _CUDA
  function torchfort_train_float_2d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :), label(:, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res =  torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                               input, input_dim, input_shape, &
                               label, label_dim, label_shape, &
                               loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_2d_dev

  function torchfort_train_double_2d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :), label(:, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_2d_dev

  function torchfort_train_float_3d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :, :), label(:, :, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_3d_dev

  function torchfort_train_double_3d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :, :), label(:, :, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_3d_dev

  function torchfort_train_float_4d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :, :, :), label(:, :, :, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_4d_dev

  function torchfort_train_double_4d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :, :, :), label(:, :, :, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_4d_dev

  function torchfort_train_float_5d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: input(:, :, :, :, :), label(:, :, :, :, :)
    real(real32) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_train_float_5d_dev

  function torchfort_train_double_5d_dev(mname, input, label, loss_val, stream) result(res)
    character(len=*) :: mname
    real(real64), device :: input(:, :, :, :, :), label(:, :, :, :, :)
    real(real64) :: loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: input_dim, label_dim

    input_dim = size(shape(input))
    label_dim = size(shape(label))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: input_shape(input_dim)
      integer(c_int64_t) :: label_shape(label_dim)

      input_shape(:) = shape(input)
      label_shape(:) = shape(label)

      res = torchfort_train_c([trim(mname) // C_NULL_CHAR], &
                              input, input_dim, input_shape, &
                              label, label_dim, label_shape, &
                              loss_val, TORCHFORT_DOUBLE, stream_)
    end block
  end function torchfort_train_double_5d_dev
#endif

  function torchfort_train_multiarg(mname, inputs, labels, loss_val, extra_loss_args, stream) result(res)
    character(len=*) :: mname
    type(torchfort_tensor_list):: inputs, labels
    real(real32) :: loss_val
    type(torchfort_tensor_list), optional :: extra_loss_args
    integer(int64), optional :: stream
    integer(c_int) :: res

    type(torchfort_tensor_list) :: extra_loss_args_
    integer(int64) :: stream_

    extra_loss_args_%member = C_NULL_PTR
    if (present(extra_loss_args)) extra_loss_args_ = extra_loss_args
    stream_ = 0
    if (present(stream)) stream_ = stream

    res =  torchfort_train_multiarg_c([trim(mname) // C_NULL_CHAR], &
                                      inputs, labels, loss_val, &
                                      extra_loss_args_, stream_)
  end function torchfort_train_multiarg

  function torchfort_save_model(mname, fname) result(res)
    character(len=*) :: mname
    character(len=*) :: fname
    integer(c_int) :: res
    res = torchfort_save_model_c([trim(mname) // C_NULL_CHAR], [trim(fname) // C_NULL_CHAR])
  end function torchfort_save_model

  function torchfort_load_model(mname, fname) result(res)
    character(len=*) :: mname
    character(len=*) :: fname
    integer(c_int) :: res
    res = torchfort_load_model_c([trim(mname) // C_NULL_CHAR], [trim(fname) // C_NULL_CHAR])
  end function torchfort_load_model

  function torchfort_save_checkpoint(mname, checkpoint_dir) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer(c_int) :: res
    res = torchfort_save_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                      [trim(checkpoint_dir) // C_NULL_CHAR])
  end function torchfort_save_checkpoint

  function torchfort_load_checkpoint_int64step(mname, checkpoint_dir, step_train, step_inference) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer(int64)  :: step_train
    integer(int64)  :: step_inference
    integer(c_int) :: res
    res = torchfort_load_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                      [trim(checkpoint_dir) // C_NULL_CHAR], &
                                      step_train, step_inference)
  end function torchfort_load_checkpoint_int64step

  function torchfort_load_checkpoint_int32step(mname, checkpoint_dir, step_train, step_inference) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer  :: step_train
    integer  :: step_inference
    integer(c_int) :: res

    integer(int64) :: step_train64, step_inference64
    step_train64 = step_train
    step_inference64 = step_inference
    res = torchfort_load_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                      [trim(checkpoint_dir) // C_NULL_CHAR], &
                                      step_train64, step_inference64)
    step_train = step_train64
    step_inference = step_inference64
  end function torchfort_load_checkpoint_int32step

  ! RL off-policy related routines
  ! logging
  function torchfort_rl_off_policy_wandb_log_int(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    integer(int32) :: val
    integer(c_int) :: res
    res = torchfort_rl_off_policy_wandb_log_int_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                  step, val)
  end function torchfort_rl_off_policy_wandb_log_int

  function torchfort_rl_off_policy_wandb_log_float(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    real(real32) :: val
    integer(c_int) :: res
    res = torchfort_rl_off_policy_wandb_log_float_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                    step, val)
  end function torchfort_rl_off_policy_wandb_log_float

  function torchfort_rl_off_policy_wandb_log_float_int32step(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int32) :: step
    real(real32) :: val
    integer(c_int) :: res
    integer(int64) :: step64
    step64 = step
    res = torchfort_rl_off_policy_wandb_log_float_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                    step64, val)
  end function torchfort_rl_off_policy_wandb_log_float_int32step

  function torchfort_rl_off_policy_wandb_log_double(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    real(real64) :: val
    integer(c_int) :: res
    res = torchfort_rl_off_policy_wandb_log_double_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                     step, val)
  end function torchfort_rl_off_policy_wandb_log_double

  function torchfort_rl_off_policy_wandb_log_double_int32step(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int32) :: step
    real(real64) :: val
    integer(c_int) :: res
    integer(int64) :: step64
    step64 = step
    res = torchfort_rl_off_policy_wandb_log_double_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                     step64, val)
  end function torchfort_rl_off_policy_wandb_log_double_int32step

  ! System creation routines
  function torchfort_rl_off_policy_create_system(sname, fname, model_dev, rb_dev) result(res)
    character(len=*) :: sname, fname
    integer(c_int) :: res
    integer(c_int) :: model_dev, rb_dev
    res = torchfort_rl_off_policy_create_system_c([trim(sname) // C_NULL_CHAR], [trim(fname) // C_NULL_CHAR], model_dev, rb_dev)
  end function torchfort_rl_off_policy_create_system

  function torchfort_rl_off_policy_create_distributed_system_MPI_F(mname, fname, comm, model_dev, rb_dev) result(res)
    character(len=*) :: mname, fname
    integer :: comm
    integer(c_int) :: model_dev, rb_dev
    integer(c_int) :: res

    type(MPI_F_Comm) :: mpi_comm_f

    mpi_comm_f%comm = comm
    res = torchfort_rl_off_policy_create_distributed_system_type(mname, fname, mpi_comm_f, model_dev, rb_dev)
  end function torchfort_rl_off_policy_create_distributed_system_MPI_F

  function torchfort_rl_off_policy_create_distributed_system_MPI_F08(mname, fname, comm, model_dev, rb_dev) result(res)
    type, bind(c) :: MPI_Comm
       integer :: MPI_VAL
    end type MPI_Comm
    character(len=*) :: mname, fname
    type(MPI_Comm) :: comm
    integer(c_int) :: model_dev, rb_dev
    integer(c_int) :: res

    type(MPI_F_Comm) :: mpi_comm_f

    mpi_comm_f%comm = comm%MPI_VAL
    res = torchfort_rl_off_policy_create_distributed_system_type(mname, fname, mpi_comm_f, model_dev, rb_dev)
  end function torchfort_rl_off_policy_create_distributed_system_MPI_F08

  function torchfort_rl_off_policy_create_distributed_system_type(mname, fname, comm, model_dev, rb_dev) result(res)
    character(len=*) :: mname, fname
    type(MPI_F_Comm) :: comm
    integer(c_int) :: model_dev, rb_dev
    integer(c_int) :: res
    type(MPI_C_Comm) :: mpi_comm_c

#ifndef MPICH
    mpi_comm_c = MPI_Comm_f2c(comm)
#else
    mpi_comm_c%comm = comm%comm
#endif
    res = torchfort_rl_off_policy_create_distributed_system_c([trim(mname) // C_NULL_CHAR], &
                                                              [trim(fname) // C_NULL_CHAR], &
                                                              mpi_comm_c, model_dev, rb_dev)
  end function torchfort_rl_off_policy_create_distributed_system_type

  ! save and load routines
  function torchfort_rl_off_policy_save_checkpoint(mname, checkpoint_dir) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer(c_int) :: res
    res = torchfort_rl_off_policy_save_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                                    [trim(checkpoint_dir) // C_NULL_CHAR])
  end function torchfort_rl_off_policy_save_checkpoint

  function torchfort_rl_off_policy_load_checkpoint(mname, checkpoint_dir) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer(c_int) :: res
    res = torchfort_rl_off_policy_load_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                                    [trim(checkpoint_dir) // C_NULL_CHAR])
  end function torchfort_rl_off_policy_load_checkpoint

  ! Training routines
  function torchfort_rl_off_policy_update_replay_buffer_float_1d_1d(mname, state_old, act_old, state_new, &
                                                                    reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state_old(:), state_new(:), act_old(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      cterminal = terminal

      res =  torchfort_rl_off_policy_update_replay_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_float_1d_1d

  function torchfort_rl_off_policy_update_replay_buffer_float_3d_3d(mname, state_old, act_old, state_new, &
                                                                    reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state_old(:, :, :), state_new(:, :, :), act_old(:, :, :)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      cterminal = terminal

      res =  torchfort_rl_off_policy_update_replay_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_float_3d_3d

  function torchfort_rl_off_policy_update_replay_buffer_float_3d_1d(mname, state_old, act_old, state_new, &
                                                                    reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state_old(:, :, :), state_new(:, :, :), act_old(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      cterminal = terminal

      res =  torchfort_rl_off_policy_update_replay_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_float_3d_1d

#ifdef _CUDA
  function torchfort_rl_off_policy_update_replay_buffer_float_1d_1d_dev(mname, state_old, act_old, state_new, &
                                                                        reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state_old(:), state_new(:), act_old(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      cterminal = terminal

      res =  torchfort_rl_off_policy_update_replay_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_float_1d_1d_dev

  function torchfort_rl_off_policy_update_replay_buffer_float_3d_3d_dev(mname, state_old, act_old, state_new, &
                                                                        reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state_old(:, :, :), state_new(:, :, :), act_old(:, :, :)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      cterminal = terminal

      res =  torchfort_rl_off_policy_update_replay_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_float_3d_3d_dev
  
  function torchfort_rl_off_policy_update_replay_buffer_float_3d_1d_dev(mname, state_old, act_old, state_new, &
                                                                        reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state_old(:, :, :), state_new(:, :, :), act_old(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      cterminal = terminal

      res =  torchfort_rl_off_policy_update_replay_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, cterminal, TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_float_3d_1d_dev
#endif

  function torchfort_rl_off_policy_update_replay_buffer_multi_float_1d_1d(mname, state_old, act_old, state_new, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state_old(:), state_new(:), act_old(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_off_policy_update_replay_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state_old, state_new, state_dim, state_shape, &
                                                                  act_old, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_multi_float_1d_1d

  function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_3d(mname, state_old, act_old, state_new, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state_old(:, :, :), state_new(:, :, :), act_old(:, :, :), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_off_policy_update_replay_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state_old, state_new, state_dim, state_shape, &
                                                                  act_old, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_3d

  function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_1d(mname, state_old, act_old, state_new, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state_old(:, :, :), state_new(:, :, :), act_old(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_off_policy_update_replay_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state_old, state_new, state_dim, state_shape, &
                                                                  act_old, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_1d

#ifdef _CUDA
  function torchfort_rl_off_policy_update_replay_buffer_multi_float_1d_1d_dev(mname, state_old, act_old, state_new, &
                                                                              reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state_old(:), state_new(:), act_old(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_off_policy_update_replay_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state_old, state_new, state_dim, state_shape, &
                                                                  act_old, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_multi_float_1d_1d_dev

  function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_3d_dev(mname, state_old, act_old, state_new, &
                                                                              reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state_old(:, :, :), state_new(:, :, :), act_old(:, :, :), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_off_policy_update_replay_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                            state_old, state_new, state_dim, state_shape, &
                                                            act_old, act_dim, act_shape, &
                                                            reward, reward_dim, reward_shape, &
                                                            terminal, terminal_dim, terminal_shape, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_3d_dev

  function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_1d_dev(mname, state_old, act_old, state_new, &
                                                                              reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state_old(:, :, :), state_new(:, :, :), act_old(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state_old))
    act_dim = size(shape(act_old))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state_old)
      act_shape(:) = shape(act_old)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_off_policy_update_replay_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state_old, state_new, state_dim, state_shape, &
                                                                  act_old, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_update_replay_buffer_multi_float_3d_1d_dev
#endif
      

  function torchfort_rl_off_policy_is_ready(mname, ready) result(res)
    character(len=*) :: mname
    logical :: ready
    integer(c_int) :: res
    res = torchfort_rl_off_policy_is_ready_c([trim(mname) // C_NULL_CHAR], ready)
  end function torchfort_rl_off_policy_is_ready

  function torchfort_rl_off_policy_train_step_float(mname, p_loss_val, q_loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32) :: p_loss_val, q_loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    stream_ = 0
    if (present(stream)) stream_ = stream

    res = torchfort_rl_off_policy_train_step_float_c([trim(mname) // C_NULL_CHAR], p_loss_val, q_loss_val, stream_)
  end function torchfort_rl_off_policy_train_step_float

  ! prediction and evaluation routines
  function torchfort_rl_off_policy_predict_explore_float_2d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                      state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, &
                                                      TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_explore_float_2d_2d

  function torchfort_rl_off_policy_predict_explore_float_4d_4d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                      state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, &
                                                      TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_explore_float_4d_4d

#ifdef _CUDA
  function torchfort_rl_off_policy_predict_explore_float_2d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                      state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, &
                                                      TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_explore_float_2d_2d_dev

  function torchfort_rl_off_policy_predict_explore_float_4d_4d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                      state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, &
                                                      TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_explore_float_4d_4d_dev
#endif

  function torchfort_rl_off_policy_predict_explore_float_4d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                      state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, &
                                                      TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_explore_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_off_policy_predict_explore_float_4d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                      state, state_dim, state_shape, &
                                                      act, act_dim, act_shape, &
                                                      TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_explore_float_4d_2d_dev
#endif

  function torchfort_rl_off_policy_predict_float_4d_4d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_float_4d_4d

#ifdef _CUDA
  function torchfort_rl_off_policy_predict_float_4d_4d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_float_4d_4d_dev
#endif

  function torchfort_rl_off_policy_predict_float_4d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_off_policy_predict_float_4d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_off_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_predict_float_4d_2d_dev
#endif

  function torchfort_rl_off_policy_evaluate_float_2d_2d(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_off_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                               state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_evaluate_float_2d_2d

  function torchfort_rl_off_policy_evaluate_float_4d_4d(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :, :, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_off_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                               state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_evaluate_float_4d_4d

#ifdef _CUDA
  function torchfort_rl_off_policy_evaluate_float_2d_2d_dev(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_off_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                               state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_evaluate_float_2d_2d_dev

  function torchfort_rl_off_policy_evaluate_float_4d_4d_dev(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :, :, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_off_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                               state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_evaluate_float_4d_4d_dev
#endif

  function torchfort_rl_off_policy_evaluate_float_4d_2d(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_off_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                               state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_evaluate_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_off_policy_evaluate_float_4d_2d_dev(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_off_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                               state, state_dim, state_shape, &
                                               act, act_dim, act_shape, &
                                               reward, reward_dim, reward_shape, &
                                               TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_off_policy_evaluate_float_4d_2d_dev
#endif

  ! RL on-policy related routines
  ! logging
  function torchfort_rl_on_policy_wandb_log_int(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    integer(int32) :: val
    integer(c_int) :: res
    res = torchfort_rl_on_policy_wandb_log_int_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                 step, val)
  end function torchfort_rl_on_policy_wandb_log_int

  function torchfort_rl_on_policy_wandb_log_float(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    real(real32) :: val
    integer(c_int) :: res
    res = torchfort_rl_on_policy_wandb_log_float_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                    step, val)
  end function torchfort_rl_on_policy_wandb_log_float

  function torchfort_rl_on_policy_wandb_log_float_int32step(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int32) :: step
    real(real32) :: val
    integer(c_int) :: res
    integer(int64) :: step64
    step64 = step
    res = torchfort_rl_on_policy_wandb_log_float_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                    step64, val)
  end function torchfort_rl_on_policy_wandb_log_float_int32step

  function torchfort_rl_on_policy_wandb_log_double(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int64) :: step
    real(real64) :: val
    integer(c_int) :: res
    res = torchfort_rl_on_policy_wandb_log_double_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                     step, val)
  end function torchfort_rl_on_policy_wandb_log_double

  function torchfort_rl_on_policy_wandb_log_double_int32step(mname, metric_name, step, val) result(res)
    character(len=*) :: mname, metric_name
    integer(int32) :: step
    real(real64) :: val
    integer(c_int) :: res
    integer(int64) :: step64
    step64 = step
    res = torchfort_rl_on_policy_wandb_log_double_c([trim(mname) // C_NULL_CHAR], [trim(metric_name) // C_NULL_CHAR], &
                                                     step64, val)
  end function torchfort_rl_on_policy_wandb_log_double_int32step

  ! System creation routines
  function torchfort_rl_on_policy_create_system(sname, fname, model_dev, rb_dev) result(res)
    character(len=*) :: sname, fname
    integer(c_int) :: res
    integer(c_int) :: model_dev, rb_dev
    res = torchfort_rl_on_policy_create_system_c([trim(sname) // C_NULL_CHAR], [trim(fname) // C_NULL_CHAR], model_dev, rb_dev)
  end function torchfort_rl_on_policy_create_system

  function torchfort_rl_on_policy_create_distributed_system_MPI_F(mname, fname, comm, model_dev, rb_dev) result(res)
    character(len=*) :: mname, fname
    integer :: comm
    integer(c_int) :: model_dev, rb_dev
    integer(c_int) :: res

    type(MPI_F_Comm) :: mpi_comm_f

    mpi_comm_f%comm = comm
    res = torchfort_rl_on_policy_create_distributed_system_type(mname, fname, mpi_comm_f, model_dev, rb_dev)
  end function torchfort_rl_on_policy_create_distributed_system_MPI_F

  function torchfort_rl_on_policy_create_distributed_system_MPI_F08(mname, fname, comm, model_dev, rb_dev) result(res)
    type, bind(c) :: MPI_Comm
       integer :: MPI_VAL
    end type MPI_Comm
    character(len=*) :: mname, fname
    type(MPI_Comm) :: comm
    integer(c_int) :: model_dev, rb_dev
    integer(c_int) :: res

    type(MPI_F_Comm) :: mpi_comm_f

    mpi_comm_f%comm = comm%MPI_VAL
    res = torchfort_rl_on_policy_create_distributed_system_type(mname, fname, mpi_comm_f, model_dev, rb_dev)
  end function torchfort_rl_on_policy_create_distributed_system_MPI_F08

  function torchfort_rl_on_policy_create_distributed_system_type(mname, fname, comm, model_dev, rb_dev) result(res)
    character(len=*) :: mname, fname
    type(MPI_F_Comm) :: comm
    integer(c_int) :: model_dev, rb_dev
    integer(c_int) :: res
    type(MPI_C_Comm) :: mpi_comm_c

#ifndef MPICH
    mpi_comm_c = MPI_Comm_f2c(comm)
#else
    mpi_comm_c%comm = comm%comm
#endif
    res = torchfort_rl_on_policy_create_distributed_system_c([trim(mname) // C_NULL_CHAR], &
                                                             [trim(fname) // C_NULL_CHAR], &
                                                             mpi_comm_c, model_dev, rb_dev)
  end function torchfort_rl_on_policy_create_distributed_system_type

  ! save and load routines
  function torchfort_rl_on_policy_save_checkpoint(mname, checkpoint_dir) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer(c_int) :: res
    res = torchfort_rl_on_policy_save_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                                    [trim(checkpoint_dir) // C_NULL_CHAR])
  end function torchfort_rl_on_policy_save_checkpoint

  function torchfort_rl_on_policy_load_checkpoint(mname, checkpoint_dir) result(res)
    character(len=*) :: mname
    character(len=*) :: checkpoint_dir
    integer(c_int) :: res
    res = torchfort_rl_on_policy_load_checkpoint_c([trim(mname) // C_NULL_CHAR], &
                                                    [trim(checkpoint_dir) // C_NULL_CHAR])
  end function torchfort_rl_on_policy_load_checkpoint

  ! Training routines
  ! single env tollout buffer updates
  function torchfort_rl_on_policy_update_rollout_buffer_float_1d_1d(mname, state, act, &
                                                                    reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:), act(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      cterminal = terminal

      res =  torchfort_rl_on_policy_update_rollout_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_float_1d_1d

  function torchfort_rl_on_policy_update_rollout_buffer_float_3d_3d(mname, state, act, &
                                                                    reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :), act(:, :, :)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      cterminal = terminal

      res =  torchfort_rl_on_policy_update_rollout_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_float_3d_3d

  function torchfort_rl_on_policy_update_rollout_buffer_float_3d_1d(mname, state, act, &
                                                                    reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :), act(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      cterminal = terminal

      res =  torchfort_rl_on_policy_update_rollout_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_float_3d_1d

#ifdef _CUDA
  function torchfort_rl_on_policy_update_rollout_buffer_float_1d_1d_dev(mname, state, act, &
                                                                        reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:), act(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      cterminal = terminal

      res =  torchfort_rl_on_policy_update_rollout_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_float_1d_1d_dev

  function torchfort_rl_on_policy_update_rollout_buffer_float_3d_3d_dev(mname, state, act, &
                                                                        reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :), act(:, :, :)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      cterminal = terminal

      res =  torchfort_rl_on_policy_update_rollout_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_float_3d_3d_dev

  function torchfort_rl_on_policy_update_rollout_buffer_float_3d_1d_dev(mname, state, act, &
                                                                        reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :), act(:)
    real(real32) :: reward
    logical :: terminal
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      logical(c_bool) :: cterminal

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      cterminal = terminal

      res =  torchfort_rl_on_policy_update_rollout_buffer_c([trim(mname) // C_NULL_CHAR], &
                                                            state, state_dim, state_shape, &
                                                            act, act_dim, act_shape, &
                                                            reward, cterminal, &
                                                            TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_float_3d_1d_dev
#endif

  
  ! multi env tollout buffer updates
  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_1d_1d(mname, state, act, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:), act(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))
	
    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_1d_1d

  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_3d(mname, state, act, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :), act(:, :, :), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_3d

  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_1d(mname, state, act, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :), act(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_1d

  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_4d_2d(mname, state, act, &
                                                                          reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_1d_1d_dev(mname, state, act, &
                                                                              reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:), act(:), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_1d_1d_dev

  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_3d_dev(mname, state, act, &
                                                                              reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :), act(:, :, :), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_3d_dev

  function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_1d_dev(mname, state, act, &
                                                                              reward, terminal, stream) result(res)
      character(len=*) :: mname
      real(real32), device :: state(:, :, :), act(:), reward(:), terminal(:)
      integer(int64), optional :: stream
      integer(c_int) :: res

      integer(int64) :: stream_

      integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
      state_dim = size(shape(state))
      act_dim = size(shape(act))
      reward_dim = size(shape(reward))
      terminal_dim = size(shape(terminal))

      stream_ = 0
      if (present(stream)) stream_ = stream

      block
        integer(c_int64_t) :: state_shape(state_dim)
        integer(c_int64_t) :: act_shape(act_dim)
        integer(c_int64_t) :: reward_shape(reward_dim)
        integer(c_int64_t) :: terminal_shape(terminal_dim)

        state_shape(:) = shape(state)
        act_shape(:) = shape(act)
        reward_shape(:) = shape(reward)
        terminal_shape(:) = shape(terminal)

        res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                    state, state_dim, state_shape, &
                                                                    act, act_dim, act_shape, &
                                                                    reward, reward_dim, reward_shape, &
                                                                    terminal, terminal_dim, terminal_shape, &
                                                                    TORCHFORT_FLOAT, stream_)
      end block
    end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_3d_1d_dev

    function torchfort_rl_on_policy_update_rollout_buffer_multi_float_4d_2d_dev(mname, state, act, &
                                                                                reward, terminal, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :), reward(:), terminal(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim, terminal_dim
    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))
    terminal_dim = size(shape(terminal))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)
      integer(c_int64_t) :: terminal_shape(terminal_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)
      terminal_shape(:) = shape(terminal)

      res =  torchfort_rl_on_policy_update_rollout_buffer_multi_c([trim(mname) // C_NULL_CHAR], &
                                                                  state, state_dim, state_shape, &
                                                                  act, act_dim, act_shape, &
                                                                  reward, reward_dim, reward_shape, &
                                                                  terminal, terminal_dim, terminal_shape, &
                                                                  TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_update_rollout_buffer_multi_float_4d_2d_dev
#endif

  ! utility routines
  function torchfort_rl_on_policy_reset_rollout_buffer(mname) result(res)
    character(len=*) :: mname
    integer(c_int) :: res
    res = torchfort_rl_on_policy_reset_rollout_buffer_c([trim(mname) // C_NULL_CHAR])
  end function torchfort_rl_on_policy_reset_rollout_buffer

  function torchfort_rl_on_policy_is_ready(mname, ready) result(res)
    character(len=*) :: mname
    logical :: ready
    integer(c_int) :: res

    res = torchfort_rl_on_policy_is_ready_c([trim(mname) // C_NULL_CHAR], ready)
  end function torchfort_rl_on_policy_is_ready

  function torchfort_rl_on_policy_train_step_float(mname, p_loss_val, q_loss_val, stream) result(res)
    character(len=*) :: mname
    real(real32) :: p_loss_val, q_loss_val
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    stream_ = 0
    if (present(stream)) stream_ = stream

    res = torchfort_rl_on_policy_train_step_float_c([trim(mname) // C_NULL_CHAR], p_loss_val, q_loss_val, stream_)
  end function torchfort_rl_on_policy_train_step_float

  ! prediction and evaluation routines
  function torchfort_rl_on_policy_predict_explore_float_2d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                     state, state_dim, state_shape, &
                                                     act, act_dim, act_shape, &
                                                     TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_explore_float_2d_2d

  function torchfort_rl_on_policy_predict_explore_float_4d_4d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                     state, state_dim, state_shape, &
                                                     act, act_dim, act_shape, &
                                                     TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_explore_float_4d_4d

#ifdef _CUDA
  function torchfort_rl_on_policy_predict_explore_float_2d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                     state, state_dim, state_shape, &
                                                     act, act_dim, act_shape, &
                                                     TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_explore_float_2d_2d_dev

  function torchfort_rl_on_policy_predict_explore_float_4d_4d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                     state, state_dim, state_shape, &
                                                     act, act_dim, act_shape, &
                                                     TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_explore_float_4d_4d_dev
#endif

  function torchfort_rl_on_policy_predict_explore_float_4d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                     state, state_dim, state_shape, &
                                                     act, act_dim, act_shape, &
                                                     TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_explore_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_on_policy_predict_explore_float_4d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_explore_c([trim(mname) // C_NULL_CHAR], &
                                                     state, state_dim, state_shape, &
                                                     act, act_dim, act_shape, &
                                                     TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_explore_float_4d_2d_dev
#endif

  function torchfort_rl_on_policy_predict_float_2d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                             state, state_dim, state_shape, &
                                             act, act_dim, act_shape, &
                                             TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_float_2d_2d

  function torchfort_rl_on_policy_predict_float_4d_4d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                             state, state_dim, state_shape, &
                                             act, act_dim, act_shape, &
                                             TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_float_4d_4d

#ifdef _CUDA
  function torchfort_rl_on_policy_predict_float_2d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                             state, state_dim, state_shape, &
                                             act, act_dim, act_shape, &
                                             TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_float_2d_2d_dev

  function torchfort_rl_on_policy_predict_float_4d_4d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :, :, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                             state, state_dim, state_shape, &
                                             act, act_dim, act_shape, &
                                             TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_float_4d_4d_dev
#endif

  function torchfort_rl_on_policy_predict_float_4d_2d(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                             state, state_dim, state_shape, &
                                             act, act_dim, act_shape, &
                                             TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_on_policy_predict_float_4d_2d_dev(mname, state, act, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)

      res = torchfort_rl_on_policy_predict_c([trim(mname) // C_NULL_CHAR], &
                                             state, state_dim, state_shape, &
                                             act, act_dim, act_shape, &
                                             TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_predict_float_4d_2d_dev
#endif

  function torchfort_rl_on_policy_evaluate_float_2d_2d(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_on_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              reward, reward_dim, reward_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_evaluate_float_2d_2d

  function torchfort_rl_on_policy_evaluate_float_4d_4d(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :, :, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_on_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              reward, reward_dim, reward_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_evaluate_float_4d_4d

  function torchfort_rl_on_policy_evaluate_float_4d_2d(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32) :: state(:, :, :, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_on_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              reward, reward_dim, reward_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_evaluate_float_4d_2d

#ifdef _CUDA
  function torchfort_rl_on_policy_evaluate_float_2d_2d_dev(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_on_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              reward, reward_dim, reward_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_evaluate_float_2d_2d_dev

  function torchfort_rl_on_policy_evaluate_float_4d_4d_dev(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :, :, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_on_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              reward, reward_dim, reward_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_evaluate_float_4d_4d_dev

  function torchfort_rl_on_policy_evaluate_float_4d_2d_dev(mname, state, act, reward, stream) result(res)
    character(len=*) :: mname
    real(real32), device :: state(:, :, :, :), act(:, :), reward(:)
    integer(int64), optional :: stream
    integer(c_int) :: res

    integer(int64) :: stream_

    integer(c_size_t) :: state_dim, act_dim, reward_dim

    state_dim = size(shape(state))
    act_dim = size(shape(act))
    reward_dim = size(shape(reward))

    stream_ = 0
    if (present(stream)) stream_ = stream

    block
      integer(c_int64_t) :: state_shape(state_dim)
      integer(c_int64_t) :: act_shape(act_dim)
      integer(c_int64_t) :: reward_shape(reward_dim)

      state_shape(:) = shape(state)
      act_shape(:) = shape(act)
      reward_shape(:) = shape(reward)

      res = torchfort_rl_on_policy_evaluate_c([trim(mname) // C_NULL_CHAR], &
                                              state, state_dim, state_shape, &
                                              act, act_dim, act_shape, &
                                              reward, reward_dim, reward_shape, &
                                              TORCHFORT_FLOAT, stream_)
    end block
  end function torchfort_rl_on_policy_evaluate_float_4d_2d_dev
#endif

  ! Tensor list management
  function torchfort_tensor_list_add_tensor_float_1d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32) :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_1d

  function torchfort_tensor_list_add_tensor_double_1d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64) :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_1d

  function torchfort_tensor_list_add_tensor_int32_1d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32) :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_1d

  function torchfort_tensor_list_add_tensor_int64_1d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64) :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_1d

  function torchfort_tensor_list_add_tensor_float_2d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32) :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_2d

  function torchfort_tensor_list_add_tensor_double_2d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64) :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_2d

  function torchfort_tensor_list_add_tensor_int32_2d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32) :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_2d

  function torchfort_tensor_list_add_tensor_int64_2d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64) :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_2d

  function torchfort_tensor_list_add_tensor_float_3d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32) :: data_arr(:, : ,:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_3d

  function torchfort_tensor_list_add_tensor_double_3d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64) :: data_arr(:, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_3d

  function torchfort_tensor_list_add_tensor_int32_3d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32) :: data_arr(:, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_3d

  function torchfort_tensor_list_add_tensor_int64_3d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64) :: data_arr(:, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_3d


  function torchfort_tensor_list_add_tensor_float_4d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32) :: data_arr(:, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_4d

  function torchfort_tensor_list_add_tensor_double_4d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64) :: data_arr(:, : ,:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_4d

  function torchfort_tensor_list_add_tensor_int32_4d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32) :: data_arr(:, : ,:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_4d

  function torchfort_tensor_list_add_tensor_int64_4d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64) :: data_arr(:, : ,:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_4d

  function torchfort_tensor_list_add_tensor_float_5d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32) :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_5d

  function torchfort_tensor_list_add_tensor_double_5d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64) :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_5d

  function torchfort_tensor_list_add_tensor_int32_5d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32) :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_5d

  function torchfort_tensor_list_add_tensor_int64_5d(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64) :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_5d

#ifdef _CUDA
  function torchfort_tensor_list_add_tensor_float_1d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32), device :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_1d_dev

  function torchfort_tensor_list_add_tensor_double_1d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64), device :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_1d_dev

  function torchfort_tensor_list_add_tensor_int32_1d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32), device :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_1d_dev

  function torchfort_tensor_list_add_tensor_int64_1d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64), device :: data_arr(:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_1d_dev

  function torchfort_tensor_list_add_tensor_float_2d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32), device :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_2d_dev

  function torchfort_tensor_list_add_tensor_double_2d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64), device :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_2d_dev

  function torchfort_tensor_list_add_tensor_int32_2d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32), device :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_2d_dev

  function torchfort_tensor_list_add_tensor_int64_2d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64), device :: data_arr(:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_2d_dev

  function torchfort_tensor_list_add_tensor_float_3d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32), device :: data_arr(:, : ,:)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_3d_dev

  function torchfort_tensor_list_add_tensor_double_3d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64), device :: data_arr(:, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_3d_dev

  function torchfort_tensor_list_add_tensor_int32_3d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32), device :: data_arr(:, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_3d_dev

  function torchfort_tensor_list_add_tensor_int64_3d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64), device :: data_arr(:, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_3d_dev

  function torchfort_tensor_list_add_tensor_float_4d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32), device :: data_arr(:, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_4d_dev

  function torchfort_tensor_list_add_tensor_double_4d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64), device :: data_arr(:, : ,:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_4d_dev

  function torchfort_tensor_list_add_tensor_int32_4d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32), device :: data_arr(:, : ,:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_4d_dev

  function torchfort_tensor_list_add_tensor_int64_4d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64), device :: data_arr(:, : ,:, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_4d_dev

  function torchfort_tensor_list_add_tensor_float_5d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real32), device :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_FLOAT)
    end block
  end function torchfort_tensor_list_add_tensor_float_5d_dev

  function torchfort_tensor_list_add_tensor_double_5d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    real(real64), device :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_DOUBLE)
    end block
  end function torchfort_tensor_list_add_tensor_double_5d_dev

  function torchfort_tensor_list_add_tensor_int32_5d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int32), device :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT32)
    end block
  end function torchfort_tensor_list_add_tensor_int32_5d_dev

  function torchfort_tensor_list_add_tensor_int64_5d_dev(tensor_list, data_arr) result(res)
    type(torchfort_tensor_list), value :: tensor_list
    integer(int64), device :: data_arr(:, :, :, :, :)
    integer(c_size_t) :: data_dim
    integer(c_int) :: res

    data_dim = size(shape(data_arr))

    block
      integer(c_int64_t) :: data_shape(data_dim)

      data_shape(:) = shape(data_arr)

      res =  torchfort_tensor_list_add_tensor_c(tensor_list, data_arr, data_dim, data_shape, &
                               TORCHFORT_INT64)
    end block
  end function torchfort_tensor_list_add_tensor_int64_5d_dev
#endif

end module torchfort
