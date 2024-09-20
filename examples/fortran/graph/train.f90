! SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

program train
  use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
  use torchfort
  implicit none

  integer :: i, istat
  integer :: num_edges, num_nodes
  integer(int64), allocatable :: edge_idx(:,:)
  real(real32), allocatable :: node_feats(:,:)
  real(real32), allocatable :: edge_feats(:,:)
  real(real32), allocatable :: node_labels(:,:)
  real(real32) :: loss_val

  type(torchfort_tensor_list) :: inputs, labels

  character(len=256) :: configfile = "config_graph.yaml"
  integer :: model_device = TORCHFORT_DEVICE_CPU

  ! Simple graph of a square
  ! y ^
  !   |
  !   3 ---- 2
  !   |      |
  !   |      |
  !   0 ---- 1 --> x

  num_edges = 4
  num_nodes = 4

  allocate(edge_idx(2, num_edges))
  allocate(node_feats(2, num_nodes))
  allocate(node_labels(2, num_nodes))
  allocate(edge_feats(3, num_nodes))

  do i = 1, num_edges
    edge_idx(1, i) = i-1
    edge_idx(2, i) = mod(i, num_edges)
  end do

  node_feats(1, :) = 1.0
  node_feats(2, :) = -1.0

  edge_feats(1, 1) = 1.0
  edge_feats(2, 1) = 0.0
  edge_feats(1, 2) = 0.0
  edge_feats(2, 2) = 1.0
  edge_feats(1, 3) = -1.0
  edge_feats(2, 3) = 0.0
  edge_feats(1, 4) = 0.0
  edge_feats(2, 4) = -1.0
  edge_feats(3, :) = 1.0

  node_labels(:,:) = 0.0

  ! setup tensor lists
  istat = torchfort_tensor_list_create(inputs)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  istat = torchfort_tensor_list_create(labels)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  istat = torchfort_tensor_list_add_tensor(inputs, edge_idx)
  istat = torchfort_tensor_list_add_tensor(inputs, node_feats)
  istat = torchfort_tensor_list_add_tensor(inputs, edge_feats)

  istat = torchfort_tensor_list_add_tensor(labels, node_labels)

  ! setup the model
  istat = torchfort_create_model("mymodel", configfile, model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  do i = 1, 1000
    istat = torchfort_train_multiarg("mymodel", inputs, labels, loss_val)
  end do

  ! destroy tensor lists
  istat = torchfort_tensor_list_destroy(inputs)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  istat = torchfort_tensor_list_destroy(labels)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

end program
