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

module utils
  use, intrinsic :: iso_fortran_env, only: real32, int64
  implicit none

  contains
    ! Advecting gaussian bump
    subroutine f_xyt(node_data, nodes, node_types, num_nodes, t)
      implicit none
      integer :: num_nodes
      real(real32) :: node_data(1, num_nodes)
      real(real32) :: nodes(2, num_nodes)
      integer(int64) :: node_types(num_nodes)
      real(real32) :: t

      integer :: i
      real(real32) :: a_x, a_y, x, y
      a_x = 1.0
      a_y = 0.0

      !$acc data copyin(nodes, node_types) copyout(node_data)
      !$acc parallel loop default(present)
      do i = 1, num_nodes
        if (node_types(i) == 0) then
          node_data(1, i) = 0.0
        else
          x = nodes(1, i)
          y = nodes(2, i)
          node_data(1, i) = exp(-20.0 * (x - a_x * t)**2) * exp(-20.0 * (y - a_y * t)**2)
        endif
      enddo
      !$acc end data

    end subroutine f_xyt

    subroutine write_node_data(node_data, num_nodes, fname)
      use, intrinsic :: iso_fortran_env, only: real32
      implicit none
      integer :: num_nodes
      real(real32) :: node_data(1, num_nodes)
      character(len=*) :: fname

      integer :: i

      !$acc update host (node_data)
      open(42, file=fname)
      do i = 1, num_nodes
        write(42, "(e12.4)") node_data(1, i)
      enddo
      close(42)

    end subroutine write_node_data
end module utils

program train
  use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
  use torchfort
  use utils
  implicit none

  integer :: i, j, k, idx, istat
  integer :: n1, n2
  integer :: num_edges, num_nodes, num_boundary_nodes
  integer(int64), allocatable :: edge_idx(:,:), node_types(:)
  real(real32), allocatable :: node_feats(:,:), node_feats_rollout(:,:)
  real(real32), allocatable :: edge_feats(:,:)
  real(real32), allocatable :: node_labels(:,:)
  real(real32) :: loss_val, mse, val
  real(real32) :: t, dt, max_t
  character(len=7) :: fidx
  character(len=256) :: filename

  integer :: num_triangles
  real(real32), allocatable:: nodes(:,:)
  integer, allocatable:: triangles(:,:)

  type(torchfort_tensor_list) :: inputs, outputs, labels
  type(torchfort_tensor_list) :: extra_loss_args

  character(len=256) :: configfile = "config.yaml"
  integer :: model_device = 0
  character(len=256) :: node_file = "nodes.txt"
  character(len=256) :: connectivity_file = "connectivity.txt"

  ! Read mesh data
  open(42, file=node_file, action='read')
  read(42, *) num_nodes
  allocate(nodes(2, num_nodes), node_types(num_nodes))
  num_boundary_nodes = 0
  do i = 1, num_nodes
    read(42, *) nodes(1, i), nodes(2, i), node_types(i) 
    if (node_types(i) == 0) num_boundary_nodes = num_boundary_nodes + 1
  end do
  close(42)

  open(42, file=connectivity_file, action='read')
  read(42, *) num_triangles
  allocate(triangles(3, num_triangles))
  do i = 1, num_triangles
    read(42, *) triangles(1, i), triangles(2, i), triangles(3, i)
  end do
  close(42)

  ! Generate edge_idx and edge_feats

  ! Collect all edges from connectivity data (bi-directional)
  num_edges = 6 * num_triangles
  allocate(edge_idx(2, num_edges))
  idx = 1
  do i = 1, num_triangles
    do k = 1, 3
      edge_idx(1, idx) = triangles(k, i)
      edge_idx(2, idx) = triangles(mod(k, 3) + 1, i)
      edge_idx(1, idx + 1) = edge_idx(2, idx)
      edge_idx(2, idx + 1) = edge_idx(1, idx)
      idx = idx + 2
    enddo
  enddo

  ! Set up edge features (dx, dy, magnitude)
  allocate(edge_feats(3, num_edges))
  do i = 1, num_edges
    n1 = edge_idx(1, i) + 1 ! connectivity data is zero-indexed, need to increment
    n2 = edge_idx(2, i) + 1
    edge_feats(1, i) = nodes(1, n2) - nodes(1, n1)
    edge_feats(2, i) = nodes(2, n2) - nodes(2, n1)
    edge_feats(3, i) = sqrt(edge_feats(1, i) * edge_feats(1, i) + edge_feats(2, i) * edge_feats(2, i))
  end do

  allocate(node_feats(1, num_nodes))
  allocate(node_labels(1, num_nodes))
  allocate(node_feats_rollout(1, num_nodes))

  !$acc data copyin(edge_idx, edge_feats, node_feats, node_labels, node_types, node_feats_rollout)

  ! Set up tensor lists
  istat = torchfort_tensor_list_create(inputs)
  istat = torchfort_tensor_list_create(labels)
  istat = torchfort_tensor_list_create(outputs)
  istat = torchfort_tensor_list_create(extra_loss_args)

  !$acc host_data use_device(edge_idx, node_feats, edge_feats, node_labels, node_types, node_feats_rollout)
  istat = torchfort_tensor_list_add_tensor(inputs, edge_idx)
  istat = torchfort_tensor_list_add_tensor(inputs, node_feats)
  istat = torchfort_tensor_list_add_tensor(inputs, edge_feats)

  istat = torchfort_tensor_list_add_tensor(labels, node_labels)

  istat = torchfort_tensor_list_add_tensor(outputs, node_feats_rollout)

  istat = torchfort_tensor_list_add_tensor(extra_loss_args, node_types)
  !$acc end host_data

  ! Set up the model
  istat = torchfort_create_model("mymodel", configfile, model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! Training
  t = 0.0
  dt = 0.1
  max_t = 5.0
  print*, "begin training..."
  do i = 1, 100000
    call f_xyt(node_feats, nodes, node_types, num_nodes, t)
    call f_xyt(node_labels, nodes, node_types, num_nodes, t + dt)
    istat = torchfort_train_multiarg("mymodel", inputs, labels, loss_val, extra_loss_args)
    t = t + dt
    if (t >= max_t) t = 0.0
  end do

  ! Rollout validation
  print*, "begin rollout validation..."
  call f_xyt(node_feats, nodes, node_types, num_nodes, 0.0)
  t = 0.0
  do i = 1, int(max_t / dt)
    istat = torchfort_inference_multiarg("mymodel", inputs, outputs)
    call f_xyt(node_feats, nodes, node_types, num_nodes, t + dt)

    ! Compute rollout MSE excluding boundary nodes
    mse = 0.0
    !$acc parallel loop reduction(+:mse) default(present)
    do j = 1, num_nodes
      val = 0.0
      if (node_types(j) /= 0) then
        val = (node_feats(1, j) - node_feats_rollout(1, j))**2
      endif
      mse = mse + val
    enddo
    write(6, '(a3,1x,f5.2,1x,a4,2x,e12.4)'), "t:", t + dt, "mse:", mse / (num_nodes - num_boundary_nodes)

    write(fidx,'(i7.7)') i
    ! Write ground truth
    filename = 'reference_'//fidx//'.txt'
    call write_node_data(node_feats, num_nodes, filename)

    ! Update input for next prediction, excluding boundary nodes
    !$acc parallel loop default(present)
    do j = 1, num_nodes
      if (node_types(j) /= 0) then
        node_feats(1, j) = node_feats_rollout(1, j)
      endif
    enddo

    ! Write prediction with preserved boundary nodes
    filename = 'prediction_'//fidx//'.txt'
    call write_node_data(node_feats, num_nodes, filename)

    t = t + dt
  enddo
  !$acc end data

  ! Destroy tensor lists
  istat = torchfort_tensor_list_destroy(inputs)
  istat = torchfort_tensor_list_destroy(labels)
  istat = torchfort_tensor_list_destroy(outputs)
  istat = torchfort_tensor_list_destroy(extra_loss_args)

end program
