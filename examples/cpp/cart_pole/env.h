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

#include <array>
#include <random>
#include <tuple>
// #include <pybind11/pybind11.h>

enum IntegratorType { EXPLICIT_EULER, SEMI_IMPLICIT_EULER };
using StateVector = std::array<float, 4>;

class CartPoleEnv {

public:
  CartPoleEnv();
  CartPoleEnv(const CartPoleEnv&) = delete;
  CartPoleEnv& operator=(const CartPoleEnv&) = delete;

  StateVector reset();
  std::pair<StateVector, StateVector> getStateBounds();

  std::tuple<StateVector, float, bool> step(float action);

private:
  // sim parameters
  bool terminated_;
  float gravity_;
  float masscart_;
  float masspole_;
  float total_mass_;
  float length_;
  float polemass_length_;
  float force_mag_;
  float dt_;
  float penalty_;
  IntegratorType kinematics_integrator_;
  int steps_beyond_terminated_;

  // threshold parameters
  float theta_threshold_radians_;
  float x_threshold_;

  // random stuff
  std::mt19937_64 rng_;
  std::uniform_real_distribution<float> uniform_dist_;

  // state vector
  StateVector state_;
};

// pybind11 stuff
// namespace py = pybind11;
// PYBIND11_MODULE(environments, m) {
//  py::class_<CartPoleEnv>(m, "CartPoleEnv", py::dynamic_attr())
//    .def(py::init<>())
//    .def("step", &CartPoleEnv::step, py::arg("action"))
//    .def("reset", &CartPoleEnv::reset);
//}
