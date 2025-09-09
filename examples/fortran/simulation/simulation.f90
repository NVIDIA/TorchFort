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

module simulation
  use, intrinsic :: iso_fortran_env, only: real32, real64
  implicit none

  real(real32), private, parameter :: PI  = 4 * atan(1.0)

  integer, private :: n, js, je
  integer, private :: simulation_device
  real(real32), private :: dx, t_total, dt, a
  real(real32), private :: ax, ay
  real(real32), private :: k = 1

  contains
    subroutine init_simulation(n_in, dt_in, a_in, t_start, rank, nranks, simulation_device_in)
      implicit none
      integer, intent(in) :: n_in, simulation_device_in
      real(real32), intent(in) :: dt_in, t_start
      real(real32), intent(in) :: a_in(2)
      integer :: rank, nranks
      n  = n_in
      dt = dt_in
      ax = a_in(1)
      ay = a_in(2)
      simulation_device = simulation_device_in

      dx = 2.0/(n)
      t_total = t_start

      js = 1
      je = n

      ! if running in parallel, split domain into slabs
      if (nranks > 1) then
        js = rank * n/nranks + 1
        je = (rank + 1) * n/nranks
      endif

    end subroutine init_simulation

    subroutine f_u(u, t)
      implicit none
      real(real32), intent(out) :: u(n, js:je)
      real(real32), intent(in) :: t
      integer :: i, j
      real(real32) ::  x, y

      !$acc parallel loop collapse(2) default(present) async if(simulation_device >= 0)
      do j = js, je
        do i = 1, n
          x = -1.0 + dx * (i-1) - mod(ax*t, 2.0)
          y = -1.0 + dx * (j-1) - mod(ay*t, 2.0)
          if (x < -1.0) x = x + 2.0
          if (y < -1.0) y = y + 2.0
          u(i, j) = sin(k*PI*x) * sin(k*PI*y)
        end do
      end do

    end subroutine f_u

    subroutine f_u_div(u_div, t)
      implicit none
      real(real32), intent(out) :: u_div(n, js:je)
      real(real32), intent(in) :: t
      integer :: i, j
      real(real32) ::  x, y

      !$acc parallel loop collapse(2) default(present) async if(simulation_device >= 0)
      do j = js, je
        do i = 1, n
          x = -1.0 + dx * (i-1) - mod(ax*t, 2.0)
          y = -1.0 + dx * (j-1) - mod(ay*t, 2.0)
          if (x < -1.0) x = x + 2.0
          if (y < -1.0) y = y + 2.0
          u_div(i, j) = k*PI * cos(k*PI*x) * sin(k*PI*y) + &
                        k*PI * sin(k*PI*x) * cos(k*PI*y)
        end do
      end do

    end subroutine f_u_div

    subroutine run_simulation_step(u, u_div)
      implicit none
      real(real32), intent(out) :: u(n, js:je)
      real(real32), intent(out) :: u_div(n, js:je)

      call f_u(u, t_total)
      call f_u_div(u_div, t_total)
      t_total = t_total + dt

    end subroutine run_simulation_step

    subroutine write_sample(sample, fname)
      character(len=*) :: fname
      real(real32), intent(in) :: sample(n, n)
      integer :: unit, i, j, err

      !$acc update host(sample) if(simulation_device >= 0)

      open(newunit=unit, file=fname, status='replace', action='write', iostat=err)
      if (err /= 0) then
        write(*,*) 'Error opening file: ', fname
        return
      endif

      do j = 1, n
        do i = 1, n-1
          write(unit, '(ES14.6E2)', advance='no') sample(i, j)
          write(unit, '(A)', advance='no') ' '
        end do
        write(unit, '(ES14.6E2)') sample(n, j)
      end do

      close(unit)
    end subroutine write_sample

end module
