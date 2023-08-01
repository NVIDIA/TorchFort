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

#include <algorithm>
#include <cmath>
#include <iostream>

#include "env.h"

static float cost_continuous_fn(float r1, float r2, float e1, float e2, float x, float x_dot, float theta,
                                float theta_dot) {
  float sign_r2 = (std::signbit(r2) ? -1.f : 1.f);
  return (sign_r2 * 100.f * r2 * r2 - 4.f * x * x) / 1000.f;
}

CartPoleEnv::CartPoleEnv() : uniform_dist_(-1.f, 1.f) {
  // important parameters
  terminated_ = true;
  gravity_ = 9.81f;
  masscart_ = 1.f;
  masspole_ = 0.1f;
  total_mass_ = masspole_ + masscart_;
  length_ = 0.5f;                         // actually half the pole's length
  polemass_length_ = masspole_ * length_; // CMS of pole
  force_mag_ = 20.f;
  dt_ = 0.02f; // seconds between state updates
  kinematics_integrator_ = EXPLICIT_EULER;
  penalty_ = 20.f;

  // threshold parameters
  theta_threshold_radians_ = 20.f * 2.f * M_PI / 360.f;
  x_threshold_ = 5.f;
}

std::pair<StateVector, StateVector> CartPoleEnv::getStateBounds() {
  return std::make_pair<StateVector, StateVector>({-2.f * x_threshold_, -1e7f, -2.f * theta_threshold_radians_, -1e7f},
                                                  {2.f * x_threshold_, 1e7f, 2.f * theta_threshold_radians_, 1e7f});
}

StateVector CartPoleEnv::reset() {

  // reset state vector
  auto gen = [&udist = this->uniform_dist_, &rng = this->rng_]() { return udist(rng); };

  std::generate(state_.begin(), state_.end(), gen);

  // now compress/expand the distribution to make sure we are inside the correct bounds
  state_[0] *= 0.3 * x_threshold_;
  state_[1] *= 0.2;
  state_[2] *= 0.3 * theta_threshold_radians_;
  state_[3] *= 0.2;

  // env info
  steps_beyond_terminated_ = -1;
  terminated_ = false;

  return state_;
}

std::tuple<StateVector, float, bool> CartPoleEnv::step(float action) {

  // extract state vectors and set force term based on action
  float x = state_[0];
  float x_dot = state_[1];
  float theta = state_[2];
  float theta_dot = state_[3];

  // action is in [-1, 1], so we multiply with force mag
  float force = action * force_mag_;

  // derived parameters
  float ctheta = std::cos(theta);
  float stheta = std::sin(theta);

  float tmp = (force + polemass_length_ * theta_dot * theta_dot * stheta) / total_mass_;
  float thetaacc =
      (gravity_ * stheta - ctheta * tmp) / (length_ * (4.0 / 3.0 - masspole_ * ctheta * ctheta / total_mass_));
  float xacc = tmp - polemass_length_ * thetaacc * ctheta / total_mass_;

  switch (kinematics_integrator_) {
  case EXPLICIT_EULER:
    x = x + dt_ * x_dot;
    x_dot = x_dot + dt_ * xacc;
    theta = theta + dt_ * theta_dot;
    theta_dot = theta_dot + dt_ * thetaacc;
    break;
  case SEMI_IMPLICIT_EULER:
    x_dot = x_dot + dt_ * xacc;
    x = x + dt_ * x_dot;
    theta_dot = theta_dot + dt_ * thetaacc;
    theta = theta + dt_ * theta_dot;
    break;
  }

  // update state
  state_ = {x, x_dot, theta, theta_dot};

  // decide if sim was terminated
  bool terminated = false;
  if ((x < -x_threshold_) || (x > x_threshold_) || (theta < -theta_threshold_radians_) ||
      (theta > theta_threshold_radians_)) {
    terminated = true;
  }

  // compute reward
  float r1 = (x_threshold_ - std::abs(x)) / x_threshold_;
  float r2 = (theta_threshold_radians_ / 4. - std::abs(theta)) / (theta_threshold_radians_ / 4.);
  float e1 = (std::abs(x)) / x_threshold_;
  float e2 = (std::abs(theta)) / theta_threshold_radians_;
  float reward = cost_continuous_fn(r1, r2, e1, e2, x, x_dot, theta, theta_dot);

  // add reward penalty for failure
  if (terminated) {
    reward -= penalty_;
  }

  return std::make_tuple(state_, reward, terminated);
}
