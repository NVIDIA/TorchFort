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
