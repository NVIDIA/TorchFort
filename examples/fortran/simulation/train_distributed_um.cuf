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

subroutine print_help_message
  print*, &
  "Usage: train_distributed_um [options]\n"// &
  "options:\n"// &
  "\t--tuning\n" //&
  "\t\tEnable unified memory tuning. (default: disabled) \n" // &
  "\t--configfile\n" // &
  "\t\tTorchFort configuration file to use. (default: config_mlp_native.yaml) \n" // &
  "\t--simulation_device\n" // &
  "\t\tDevice to run simulation on. (-1 for CPU, 0 for GPU. default: GPU) \n" // &
  "\t--train_device\n" // &
  "\t\tDevice to run model training/inference on. (-1 for CPU, 0 for GPU. default: GPU) \n" // &
  "\t--ntrain_steps\n" // &
  "\t\tNumber of training steps to run. (default: 100000) \n" // &
  "\t--nval_steps\n" // &
  "\t\tNumber of validation steps to run. (default: 1000) \n" // &
  "\t--val_write_freq\n" // &
  "\t\tFrequency to write validation HDF5 files. (default: 10) \n" // &
  "\t--checkpoint_dir\n" // &
  "\t\tCheckpoint directory to load. (default: don't load checkpoint) \n" // &
  "\t--output_model_name\n" // &
  "\t\tFilename for saved model. (default: model.pt) \n" // &
  "\t--output_checkpoint_dir\n" // &
  "\t\tName of checkpoint directory to save. (default: checkpoint) \n"
end subroutine print_help_message

program train_distributed_um
    use, intrinsic :: iso_fortran_env, only: real32, real64
#ifdef _OPENACC
  use openacc
#endif
  use mpi
  use simulation
  use torchfort
  use cudafor
  implicit none

  logical :: tuning = .false.
  integer :: i, j, istat
  integer :: n, nchannels, batch_size
  real(real32) :: a(2), dt
  real(real32) :: loss_val
  real(real64) :: mse
  real(real32), managed, dimension(:,:), allocatable :: u, u_div
  real(real32), managed, dimension(:,:,:,:), allocatable :: input, label, output
  real(real32), managed, dimension(:,:,:,:), allocatable :: input_local, label_local
  character(len=7) :: idx
  character(len=256) :: filename
  logical :: load_ckpt = .false.
  integer :: train_step_ckpt = 0
  integer :: val_step_ckpt = 0

  integer :: rank, local_rank, nranks
  integer :: local_comm
#ifdef _OPENACC
  integer(acc_device_kind) :: dev_type
#endif
  integer, allocatable :: sendcounts(:), recvcounts(:)
  integer, allocatable :: sdispls(:), rdispls(:)

  ! command line arguments
  character(len=256) :: configfile = "config_mlp_native.yaml"
  character(len=256) :: checkpoint_dir
  character(len=256) :: output_model_name = "model.pt"
  character(len=256) :: output_checkpoint_dir = "checkpoint"
  integer :: ntrain_steps = 100000
  integer :: nval_steps = 1000
  integer :: val_write_freq = 10
  integer :: model_device = 0
  integer :: simulation_device = 0

  logical :: skip_next
  character(len=256) :: arg

  ! initialize MPI
  call MPI_Init(istat)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, istat)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, istat)
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, istat)
  call MPI_Comm_rank(local_comm, local_rank, istat)

  if (nranks /= 2) then
    print*, "This example requires 2 ranks to run. Exiting."
    stop
  endif

  ! read command line arguments
  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--tuning')
        tuning = .true.
      case('--configfile')
        call get_command_argument(i+1, arg)
        read(arg, *) configfile
        skip_next = .true.
      case('--checkpoint_dir')
        call get_command_argument(i+1, arg)
        read(arg, *) checkpoint_dir
        skip_next = .true.
        load_ckpt = .true.
      case('--output_model_name')
        call get_command_argument(i+1, arg)
        read(arg, *) output_model_name
        skip_next = .true.
      case('--output_checkpoint_dir')
        call get_command_argument(i+1, arg)
        read(arg, *) output_checkpoint_dir
        skip_next = .true.
      case('--ntrain_steps')
        call get_command_argument(i+1, arg)
        read(arg, *) ntrain_steps
        skip_next = .true.
      case('--nval_steps')
        call get_command_argument(i+1, arg)
        read(arg, *) nval_steps
        skip_next = .true.
      case('--val_write_freq')
        call get_command_argument(i+1, arg)
        read(arg, *) val_write_freq
        skip_next = .true.
      case('--train_device')
        call get_command_argument(i+1, arg)
        read(arg, *) model_device
        if (model_device /= -1 .and. model_device /= 0) then
          print*, "Invalid train device type argument."
          call exit(1)
        endif
        skip_next = .true.
      case('--simulation_device')
        call get_command_argument(i+1, arg)
        read(arg, *) simulation_device
        if (simulation_device /= -1 .and. simulation_device /= 0) then
          print*, "Invalid simulation device type argument."
          call exit(1)
        endif
        skip_next = .true.
      case('-h')
        if (rank == 0) call print_help_message
        call MPI_Finalize(istat)
        call exit(0)
      case default
        print*, "Unknown argument."
        call exit(1)
    end select
  end do

#ifndef _OPENACC
  if (simulation_device /= -1) then
    print*, "OpenACC support required to run simulation on GPU. &
             Set --simulation_device -1 to run simulation on CPU."
    call exit(1)
  endif
#endif
#ifdef _OPENACC
  if (simulation_device == 0) then
    ! assign GPUs by local rank
    dev_type = acc_get_device_type()
    call acc_set_device_num(local_rank, dev_type)
    call acc_init(dev_type)
  endif
#endif
  if (model_device == 0) then
    ! assign GPUs by local rank
    model_device = local_rank
  endif


  if (rank == 0) then
    print*, "Run settings:"
    print*, "\tconfigfile: ", trim(configfile)
    if (simulation_device == TORCHFORT_DEVICE_CPU) then
      print*, "\tsimulation_device: cpu"
    else
      print*, "\tsimulation_device: gpu"
    endif
    if (model_device == TORCHFORT_DEVICE_CPU) then
      print*, "\ttrain_device: cpu"
    else
      print*, "\ttrain_device: gpu"
    endif
    if (load_ckpt) then
      print*, "\tcheckpoint_dir: ", trim(checkpoint_dir)
    else
      print*, "\tcheckpoint_dir:", "NONE"
    endif
    print*, "\toutput_model_name: ", trim(output_model_name)
    print*, "\toutput_checkpoint_dir: ", trim(output_checkpoint_dir)
    print*, "\tntrain_steps: ", ntrain_steps
    print*, "\tnval_steps: ", nval_steps
    print*, "\tval_write_freq: ", val_write_freq
    print*
  endif

  ! model/simulation parameters
  n = 32
  nchannels = 1
  batch_size = 16 / nranks ! splitting global batch across GPUs
  dt = 0.01
  a = [1.0, 0.789] ! off-angle to generate more varied training data

  ! allocate "simulation" data sized for *local* domain
  allocate(u(n, n/nranks))
  allocate(u_div(n, n/nranks))

  ! allocate training/inference data in standard 2D layout (NCHW, row-major),
  ! sized for *global* domain
  allocate(input_local(n, n/nranks, nchannels, batch_size*nranks))
  allocate(label_local(n, n/nranks, nchannels, batch_size*nranks))
  allocate(input(n, n, nchannels, batch_size))
  allocate(label(n, n, nchannels, batch_size))
  allocate(output(n, n, nchannels, batch_size))

  ! allocate and set up arrays for MPI Alltoallv (batch redistribution)
  allocate(sendcounts(nranks), recvcounts(nranks))
  allocate(sdispls(nranks), rdispls(nranks))
  do i = 1, nranks
    sendcounts(i) = n * n/nranks
    recvcounts(i) = n * n/nranks
  end do
  sdispls(1) = 0
  rdispls(1) = 0
  do i = 2, nranks
    sdispls(i) = sdispls(i-1) + n*n/nranks*batch_size
    rdispls(i) = rdispls(i-1) + n*n/nranks
  end do

  ! set torch benchmark mode
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! setup the data parallel model
  istat = torchfort_create_distributed_model("mymodel", configfile, MPI_COMM_WORLD, model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! load training checkpoint if requested
  if (load_ckpt) then
    if (rank == 0) print*, "loading checkpoint..."
    istat = torchfort_load_checkpoint("mymodel", checkpoint_dir, train_step_ckpt, val_step_ckpt)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  endif

  call init_simulation(n, dt, a, train_step_ckpt*batch_size*dt, rank, nranks, simulation_device)

  ! run training
  if (rank == 0 .and. ntrain_steps >= 1) print*, "start training..."

  do i = 1, ntrain_steps
    do j = 1, batch_size * nranks
      call run_simulation_step(u, u_div)
      !$acc kernels if(simulation_device >= 0) async
      input_local(:,:,1,j) = u
      label_local(:,:,1,j) = u_div
      !$acc end kernels
    end do

    !$acc wait

    ! distribute local batch data across GPUs for data parallel training
    do j = 1, batch_size

      call MPI_Alltoallv(input_local(:,:,1,j), sendcounts, sdispls, MPI_FLOAT, &
                         input(:,:,1,j), recvcounts, rdispls, MPI_FLOAT, &
                         MPI_COMM_WORLD, istat)
      call MPI_Alltoallv(label_local(:,:,1,j), sendcounts, sdispls, MPI_FLOAT, &
                         label(:,:,1,j), recvcounts, rdispls, MPI_FLOAT, &
                         MPI_COMM_WORLD, istat)

    end do

    !$acc wait

    istat = torchfort_train("mymodel", input, label, loss_val)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    !$acc wait
  end do

  if (rank == 0 .and. ntrain_steps >= 1) print*, "final training loss: ", loss_val

  ! run inference
  if (rank == 0 .and. nval_steps >= 1) print*, "start validation..."

  do i = 1, nval_steps
    call run_simulation_step(u, u_div)
    !$acc kernels async if(simulation_device >= 0)
    input_local(:,:,1,1) = u
    label_local(:,:,1,1) = u_div
    !$acc end kernels

    !$acc wait

    ! gather sample on all GPUs

    call MPI_Allgather(input_local(:,:,1,1), n * n/nranks, MPI_FLOAT, &
                       input(:,:,1,1), n * n/nranks, MPI_FLOAT, &
                       MPI_COMM_WORLD, istat)
    call MPI_Allgather(label_local(:,:,1,1), n * n/nranks, MPI_FLOAT, &
                       label(:,:,1,1), n * n/nranks, MPI_FLOAT, &
                       MPI_COMM_WORLD, istat)


    !$acc wait

    istat = torchfort_inference("mymodel", input(:,:,1:1,1:1), output(:,:,1:1,1:1))
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    !$acc wait

    !$acc kernels if(simulation_device >= 0)
    mse = sum((label(:,:,1,1) - output(:,:,1,1))**2) / (n*n)
    !$acc end kernels

    if (rank == 0 .and. mod(i-1, val_write_freq) == 0) then
      print*, "writing validation sample:", i, "mse:", mse
      write(idx,'(i7.7)') i
      filename = 'input_'//idx//'.h5'
      call write_sample(input(:,:,1,1), filename)
      filename = 'label_'//idx//'.h5'
      call write_sample(label(:,:,1,1), filename)
      filename = 'output_'//idx//'.h5'
      call write_sample(output(:,:,1,1), filename)
    endif
  end do


  if (rank == 0) then
    print*, "saving model and writing checkpoint..."
    istat = torchfort_save_model("mymodel", output_model_name)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_save_checkpoint("mymodel", output_checkpoint_dir)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  endif

  call MPI_Finalize(istat)

end program train_distributed_um
