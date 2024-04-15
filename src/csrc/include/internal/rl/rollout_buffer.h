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

#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/rl/rl.h"

namespace torchfort {

namespace rl {

using BufferEntry = std::tuple<torch::Tensor, torch::Tensor, float, float, float, bool>;

enum RewardReductionMode { Sum = 1, Mean = 2, WeightedMean = 3, SumNoSkip = 4, MeanNoSkip = 5, WeightedMeanNoSkip = 6 };

// abstract base class for rollout buffer
class RolloutBuffer {
public:
  // disable copy constructor
  RolloutBuffer(const RolloutBuffer&) = delete;
  // base constructor
  RolloutBuffer(size_t size, torchfort_device_t device) : size_(size), device_(get_device(device)) {}

  // virtual functions
  virtual void update(torch::Tensor, torch::Tensor, float, float, float, bool) = 0;
  virtual void finalizeTrajectory(float, bool) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  sample(int) = 0;
  virtual bool isReady() const = 0;
  virtual void reset() = 0;
  virtual void printInfo() const = 0;
  virtual void save(const std::string& fname) const = 0;
  virtual void load(const std::string& fname) = 0;
  virtual torch::Device device() const = 0;

protected:
  size_t size_;
  torch::Device device_;
};

class GAELambdaRolloutBuffer : public RolloutBuffer, public std::enable_shared_from_this<RolloutBuffer> {

public:
  // constructor
  GAELambdaRolloutBuffer(size_t size, float gamma, float lambda, torchfort_device_t device)
    : RolloutBuffer(size, device), finalized_(false), gamma_(gamma), lambda_(lambda), rng_(), returns_(size), advantages_(size) {}

  // disable copy constructor
  GAELambdaRolloutBuffer(const GAELambdaRolloutBuffer&) = delete;

  // update
  void update(torch::Tensor s, torch::Tensor a, float r, float q, float log_p, bool e) {

    // add no grad guard
    torch::NoGradGuard no_grad;

    // clone the tensors and move to device
    auto sc = s.to(device_, s.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto ac = a.to(device_, a.dtype(), /* non_blocking = */ false, /* copy = */ true);

    // add the newest element in front
    buffer_.push_front(std::make_tuple(sc, ac, r, q, log_p, e));

    // if we reached max size already, remove the oldest element
    // this ensures that we never have a buffer which has more elements than expected
    // the user should ensure that he queries is_ready frequently and performs
    // train steps accordingly
    if (buffer_.size() > size_) {
      buffer_.pop_back();
    }
  }

  // compute returns and advantages
  void finalizeTrajectory(float last_value, bool done) {
    float last_gae_lam = 0.;

    // initialize starting values
    float next_non_terminal = 1.0 - float(done);
    float next_value = last_value;

    // get first buffer entry
    torch::Tensor s, a;
    float r, q, log_p;
    bool e;
    std::tie(s, a, r, q, log_p, e) = buffer_.at(size_ - 1);
    float delta = r + gamma_ * next_value * next_non_terminal - q;
    last_gae_lam = delta + gamma_ * lambda_ * next_non_terminal * last_gae_lam;
    advantages_[size_ - 1] = last_gae_lam;
    returns_[size_ - 1] = last_gae_lam + q;
    // we need to keep track of those
    next_value = q;
    next_non_terminal = 1.0 - float(e);
    for (size_t step = size_ - 2; step >= 0; step--) {
      std::tie(s, a, r, q, log_p, e) = buffer_.at(step);
      delta = r + gamma_ * next_value * next_non_terminal - q;
      last_gae_lam = delta + gamma_ * lambda_ * next_non_terminal * last_gae_lam;
      advantages_[step] = last_gae_lam;
      returns_[step] = last_gae_lam + q;
      next_value = q;
      next_non_terminal = 1.0 - float(e);
    }
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
    auto stens_list = std::vector<torch::Tensor>(batch_size);
    auto atens_list = std::vector<torch::Tensor>(batch_size);
    auto q_list = std::vector<float>(batch_size);
    auto log_p_list = std::vector<float>(batch_size);
    auto adv_list = std::vector<float>(batch_size);
    auto ret_list = std::vector<float>(batch_size);

    // we need to exclude the last element here, since it is a closed interval
    std::uniform_int_distribution<size_t> uniform_dist(0, size_ - 1);
    for (int sample = 0; sample < batch_size; sample++) {
      auto index = uniform_dist(rng_);

      // get buffer entry
      float r;
      bool e;
      std::tie(stens_list[sample], atens_list[sample], r, q_list[sample], log_p_list[sample], e) = buffer_.at(index);
      adv_list[sample] = advantages_[sample];
      ret_list[sample] = returns_[sample];
    }
    
    // stack the lists
    auto stens = torch::stack(stens_list, 0);
    auto atens = torch::stack(atens_list, 0);

    // create new tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto qtens = torch::from_blob(q_list.data(), {batch_size, 1}, options).clone();
    auto logptens = torch::from_blob(log_p_list.data(), {batch_size, 1}, options).clone();
    auto advtens = torch::from_blob(adv_list.data(), {batch_size, 1}, options).clone();
    auto rettens = torch::from_blob(ret_list.data(), {batch_size, 1}, options).clone();

    return std::make_tuple(stens, atens, qtens, logptens, advtens, rettens);
  }

  // check functions
  bool isReady() const { return (buffer_.size() == size_); }

  // reset the buffer
  void reset() {
    buffer_.clear();

    // zero out the returns and advantage vectors just to be safe
    std::fill(advantages_.begin(), advantages_.end(), 0.);
    std::fill(returns_.begin(), returns_.end(), 0.);

    // finally, set the finalized flag to false
    finalized_ = false;
    
    return;
  }

  void save(const std::string& fname) const {
    // create an ordered dict with the buffer contents:
    std::vector<torch::Tensor> s_data, a_data;
    std::vector<torch::Tensor> r_data, q_data, log_p_data, e_data;
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
    }

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

    return;
  }

  void load(const std::string& fname) {
    // get vectors for buffers:
    std::vector<torch::Tensor> s_data, a_data;
    std::vector<torch::Tensor> r_data, q_data, log_p_data, e_data;

    using namespace torchfort;
    std::filesystem::path root_dir(fname);

    torch::load(s_data, root_dir / "s_data.pt");
    torch::load(a_data, root_dir / "a_data.pt");
    torch::load(r_data, root_dir / "r_data.pt");
    torch::load(q_data, root_dir / "q_data.pt");
    torch::load(log_p_data, root_dir / "log_p_data.pt");
    torch::load(e_data, root_dir / "e_data.pt");

    // iterate over loaded data and populate buffer
    buffer_.clear();
    for (size_t index = 0; index < s_data.size(); ++index) {
      auto s = s_data[index];
      auto a = a_data[index];
      float r = r_data[index].item<float>();
      float q = q_data[index].item<float>();
      float log_p = log_p_data[index].item<float>();
      bool e = e_data[index].item<bool>();

      // update buffer
      this->update(s, a, r, q, log_p, e);
    }
  }

  void printInfo() const {
    std::cout << "GAE-lambda rollout buffer parameters:" << std::endl;
    std::cout << "size = " << size_ << std::endl;
    std::cout << "gamma = " << gamma_ << std::endl;
    std::cout << "lambda = " << lambda_ << std::endl;
  }

  torch::Device device() const {
    return device_;
  }

private:
  // keep track of whether buffer is finalized_:
  bool finalized_;
  // we need the discount factors
  float gamma_, lambda_;
  // the rbuffer contains tuples: (s, a, r, q, log_p, e)
  std::deque<BufferEntry> buffer_;
  // additional storage for returns and advantage
  std::vector<float> returns_, advantages_;
  // rng
  std::mt19937_64 rng_;
};

} // namespace rl
} // namespace torchfort
