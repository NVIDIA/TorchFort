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
#include <deque>
#include <random>
#include <tuple>

#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/rl/rl.h"

namespace torchfort {

namespace rl {

using BufferEntry = std::tuple<torch::Tensor, torch::Tensor, float, float, float, bool>;
using ExtendedBufferEntry = std::tuple<torch::Tensor, torch::Tensor, float, float, float, float, float, bool>;

// abstract base class for rollout buffer
class RolloutBuffer {
public:
  // disable copy constructor
  RolloutBuffer(const RolloutBuffer&) = delete;
  // base constructor
  RolloutBuffer(size_t size, int device)
      : size_(size), device_(get_device(device)), last_episode_starts_(true), indices_(size), pos_(0), rng_() {
    // fill index vector with indices
    std::generate(indices_.begin(), indices_.end(), [n = 0]() mutable { return n++; });
  }

  // accessors
  size_t getSize() const { return size_; }

  // virtual functions
  virtual void update(torch::Tensor, torch::Tensor, float, float, float, bool) = 0;
  virtual void finalize(float, bool) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  sample(int) = 0;
  virtual BufferEntry get(int) = 0;
  virtual ExtendedBufferEntry getFull(int) = 0;
  virtual bool isReady() const = 0;
  virtual void reset() = 0;
  virtual void printInfo() const = 0;
  virtual void save(const std::string& fname) const = 0;
  virtual void load(const std::string& fname) = 0;
  virtual torch::Device device() const = 0;

protected:
  // shuffling
  void resetIndices_() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
    pos_ = 0;
  }

  size_t size_;
  torch::Device device_;
  bool last_episode_starts_;
  std::vector<int> indices_;
  size_t pos_;
  std::mt19937_64 rng_;
};

class GAELambdaRolloutBuffer : public RolloutBuffer, public std::enable_shared_from_this<RolloutBuffer> {

public:
  // constructor
  GAELambdaRolloutBuffer(size_t size, float gamma, float lambda, int device)
      : RolloutBuffer(size, device), finalized_(false), gamma_(gamma), lambda_(lambda), returns_(size),
        advantages_(size) {}

  // disable copy constructor
  GAELambdaRolloutBuffer(const GAELambdaRolloutBuffer&) = delete;

  // update
  void update(torch::Tensor s, torch::Tensor a, float r, float q, float log_p, bool d) {

    // add no grad guard
    torch::NoGradGuard no_grad;

    // do not push anything if buffer is full, instead check if buffer is finalized
    if (buffer_.size() == size_) {
      if (!finalized_) {
        finalize(q, d);
      }
    } else {

      // clone the tensors and move to device
      auto sc = s.to(device_, s.dtype(), /* non_blocking = */ false, /* copy = */ true);
      auto ac = a.to(device_, a.dtype(), /* non_blocking = */ false, /* copy = */ true);

      // add the newest/latest element in back
      buffer_.push_back(std::make_tuple(sc, ac, r, q, log_p, last_episode_starts_));
    }

    // set next_state_inital to done:
    last_episode_starts_ = d;
  }

  // compute returns and advantages
  void finalize(float last_value, bool done) {

    // temporary variables
    torch::Tensor s, a;
    float r, q, log_p, delta;
    bool e;

    // we need to keep track of those
    // initialize starting values
    float last_gae_lam = 0.;
    float next_non_terminal = (done ? 0. : 1.);
    float next_value = last_value;

    for (int64_t step = size_ - 1; step >= 0; step--) {
      std::tie(s, a, r, q, log_p, e) = buffer_.at(step);
      delta = r + gamma_ * next_value * next_non_terminal - q;
      last_gae_lam = delta + gamma_ * lambda_ * next_non_terminal * last_gae_lam;
      advantages_[step] = last_gae_lam;
      returns_[step] = last_gae_lam + q;
      next_value = q;
      next_non_terminal = (e ? 0. : 1.);
    }

    // create shuffled index list for sampling:
    resetIndices_();

    // set finalized to true
    finalized_ = true;

    return;
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  sample(int batch_size) {

    // ensure that the buffer is finalized:
    if (!finalized_) {
      throw std::runtime_error("Finalize the rollout buffers trajectory before you start sampling from it.");
    }

    // add no grad guard
    torch::NoGradGuard no_grad;

    // we need those
    std::vector<torch::Tensor> stens_list, atens_list;
    std::vector<float> q_list, log_p_list, adv_list, ret_list;

    // go through the buffer in random order
    auto lower = pos_;
    auto upper = std::min(pos_ + static_cast<size_t>(batch_size), size_);
    for (size_t sample = lower; sample < upper; sample++) {
      // use sampling without replacement
      auto index = indices_[sample];

      // get buffer entry
      torch::Tensor stens, atens;
      float r, q, log_p;
      bool d;
      std::tie(stens, atens, r, q, log_p, d) = buffer_.at(index);

      // append to lists
      stens_list.push_back(stens);
      atens_list.push_back(atens);
      q_list.push_back(q);
      log_p_list.push_back(log_p);
      adv_list.push_back(advantages_[index]);
      ret_list.push_back(returns_[index]);
    }

    // update buffer position
    pos_ = upper;

    // stack the lists
    auto stens = torch::stack(stens_list, 0).clone();
    auto atens = torch::stack(atens_list, 0).clone();

    // create new tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto qtens = torch::from_blob(q_list.data(), {batch_size, 1}, options).clone();
    auto logptens = torch::from_blob(log_p_list.data(), {batch_size, 1}, options).clone();
    auto advtens = torch::from_blob(adv_list.data(), {batch_size, 1}, options).clone();
    auto rettens = torch::from_blob(ret_list.data(), {batch_size, 1}, options).clone();

    // some finalization checks:
    if (pos_ >= size_) {
      // shuffle vector and reset position: this allows for continued sampling
      resetIndices_();
    }

    return std::make_tuple(stens, atens, qtens, logptens, advtens, rettens);
  }

  BufferEntry get(int index) {
    // sanity checks
    if ((index < 0) || (index >= buffer_.size())) {
      throw std::runtime_error("GAELambdaRolloutBuffer::get: index " + std::to_string(index) + " out of bounds [0, " +
                               std::to_string(buffer_.size()) + ").");
    }

    // add no grad guard
    torch::NoGradGuard no_grad;

    // emit the sample at index
    torch::Tensor stens, atens;
    float r, q, log_p;
    bool d;
    std::tie(stens, atens, r, q, log_p, d) = buffer_.at(index);

    return std::make_tuple(stens, atens, r, q, log_p, d);
  }

  ExtendedBufferEntry getFull(int index) {
    // sanity checks
    if ((index < 0) || (index >= buffer_.size())) {
      throw std::runtime_error("GAELambdaRolloutBuffer::getFull: index " + std::to_string(index) +
                               " out of bounds [0, " + std::to_string(buffer_.size()) + ").");
    }

    if (!finalized_) {
      throw std::runtime_error(
          "GAELambdaRolloutBuffer::getFull: the buffer needs to be finalized before calling getFull.");
    }

    // add no grad guard
    torch::NoGradGuard no_grad;

    // emit the sample at index
    torch::Tensor stens, atens;
    float r, q, log_p, adv, ret;
    bool d;
    std::tie(stens, atens, r, q, log_p, d) = buffer_.at(index);
    adv = advantages_[index];
    ret = returns_[index];

    return std::make_tuple(stens, atens, r, q, log_p, adv, ret, d);
  }

  // check functions
  bool isReady() const { return ((buffer_.size() == size_) && finalized_); }

  // reset the buffer
  void reset() {
    buffer_.clear();

    // zero out the returns and advantage vectors just to be safe
    std::fill(advantages_.begin(), advantages_.end(), 0.);
    std::fill(returns_.begin(), returns_.end(), 0.);

    // finally, set the finalized flag to false
    finalized_ = false;

    // mark a new episode:
    last_episode_starts_ = true;

    return;
  }

  void save(const std::string& fname) const {
    // create an ordered dict with the buffer contents:
    std::vector<torch::Tensor> s_data, a_data;
    std::vector<torch::Tensor> r_data, q_data, log_p_data, e_data, adv_data, ret_data;
    std::vector<torch::Tensor> state_data;

    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto options_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    for (size_t index = 0; index < buffer_.size(); ++index) {
      torch::Tensor s, a;
      float r, q, log_p;
      bool e;
      std::tie(s, a, r, q, log_p, e) = buffer_.at(index);
      s_data.push_back(s.to(torch::kCPU));
      a_data.push_back(a.to(torch::kCPU));

      auto rt = torch::from_blob(&r, {1}, options_f).clone();
      r_data.push_back(rt);
      auto qt = torch::from_blob(&q, {1}, options_f).clone();
      q_data.push_back(qt);
      auto log_pt = torch::from_blob(&log_p, {1}, options_f).clone();
      log_p_data.push_back(log_pt);
      auto et = torch::from_blob(&e, {1}, options_b).clone();
      e_data.push_back(et);
      float adv = advantages_[index];
      auto advt = torch::from_blob(&adv, {1}, options_f).clone();
      adv_data.push_back(advt);
      float ret = returns_[index];
      auto rett = torch::from_blob(&ret, {1}, options_f).clone();
      ret_data.push_back(rett);
    }
    // state vector
    bool tmpbool = last_episode_starts_;
    auto st = torch::from_blob(&tmpbool, {1}, options_b).clone();
    state_data.push_back(st);
    tmpbool = finalized_;
    st = torch::from_blob(&tmpbool, {1}, options_b).clone();
    state_data.push_back(st);

    // create subdirectory:
    using namespace torchfort;
    std::filesystem::path root_dir(fname);

    if (!std::filesystem::exists(root_dir)) {
      bool rv = std::filesystem::create_directory(root_dir);
      if (!rv) {
        throw std::runtime_error("Could not create directory for rollout buffer.");
      }
    }

    // save the buffer:
    torch::save(s_data, root_dir / "s_data.pt");
    torch::save(a_data, root_dir / "a_data.pt");
    torch::save(r_data, root_dir / "r_data.pt");
    torch::save(q_data, root_dir / "q_data.pt");
    torch::save(log_p_data, root_dir / "log_p_data.pt");
    torch::save(e_data, root_dir / "e_data.pt");
    torch::save(adv_data, root_dir / "adv_data.pt");
    torch::save(ret_data, root_dir / "ret_data.pt");
    torch::save(state_data, root_dir / "buffer_state.pt");

    return;
  }

  void load(const std::string& fname) {
    // get vectors for buffers:
    std::vector<torch::Tensor> s_data, a_data;
    std::vector<torch::Tensor> r_data, q_data, log_p_data, e_data, adv_data, ret_data;
    std::vector<torch::Tensor> state_data;

    using namespace torchfort;
    std::filesystem::path root_dir(fname);

    torch::load(s_data, root_dir / "s_data.pt");
    torch::load(a_data, root_dir / "a_data.pt");
    torch::load(r_data, root_dir / "r_data.pt");
    torch::load(q_data, root_dir / "q_data.pt");
    torch::load(log_p_data, root_dir / "log_p_data.pt");
    torch::load(e_data, root_dir / "e_data.pt");
    torch::load(adv_data, root_dir / "adv_data.pt");
    torch::load(ret_data, root_dir / "ret_data.pt");
    torch::load(state_data, root_dir / "buffer_state.pt");

    // iterate over loaded data and populate buffer
    buffer_.clear();
    advantages_.resize(s_data.size());
    returns_.resize(s_data.size());
    for (size_t index = 0; index < s_data.size(); ++index) {
      auto s = s_data[index];
      auto a = a_data[index];
      float r = r_data[index].item<float>();
      float q = q_data[index].item<float>();
      float log_p = log_p_data[index].item<float>();
      bool e = e_data[index].item<bool>();
      advantages_[index] = adv_data[index].item<float>();
      returns_[index] = ret_data[index].item<float>();

      // writing manual update here instead of using the member function to override the next last episode logic:
      // clone the tensors and move to device
      auto sc = s.to(device_, s.dtype(), /* non_blocking = */ false, /* copy = */ true);
      auto ac = a.to(device_, a.dtype(), /* non_blocking = */ false, /* copy = */ true);

      // add the newest/latest element in back
      buffer_.push_back(std::make_tuple(sc, ac, r, q, log_p, e));
    }

    // restore buffer state
    last_episode_starts_ = state_data[0].item<bool>();
    finalized_ = state_data[1].item<bool>();

    return;
  }

  void printInfo() const {
    std::cout << "GAE-lambda rollout buffer parameters:" << std::endl;
    std::cout << "size = " << size_ << std::endl;
    std::cout << "gamma = " << gamma_ << std::endl;
    std::cout << "lambda = " << lambda_ << std::endl;
  }

  torch::Device device() const { return device_; }

private:
  // keep track of whether buffer is finalized_:
  bool finalized_;
  // we need the discount factors
  float gamma_, lambda_;
  // the rbuffer contains tuples: (s, a, r, q, log_p, e)
  std::deque<BufferEntry> buffer_;
  // additional storage for returns and advantage
  std::vector<float> returns_, advantages_;
};

} // namespace rl
} // namespace torchfort
