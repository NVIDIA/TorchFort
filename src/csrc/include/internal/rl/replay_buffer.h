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

using BufferEntry = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>;

enum RewardReductionMode { Sum = 1, Mean = 2, WeightedMean = 3, SumNoSkip = 4, MeanNoSkip = 5, WeightedMeanNoSkip = 6 };

// abstract base class for replay buffer
class ReplayBuffer {
public:
  // disable copy constructor
  ReplayBuffer(const ReplayBuffer&) = delete;
  // base constructor
  ReplayBuffer(size_t max_size, size_t min_size, int device)
      : max_size_(max_size), min_size_(min_size), device_(get_device(device)) {}

  // accessor functions
  size_t getMinSize() const { return min_size_; }
  size_t getMaxSize() const { return max_size_; }

  // virtual functions
  virtual void update(torch::Tensor, torch::Tensor, torch::Tensor, float, bool) = 0;
  // sample element randomly
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(int) = 0;
  // get specific element
  virtual BufferEntry get(int) = 0;
  // helper functions
  virtual bool isReady() const = 0;
  virtual void reset() = 0;
  virtual size_t getSize() const = 0;
  virtual void printInfo() const = 0;
  virtual void save(const std::string& fname) const = 0;
  virtual void load(const std::string& fname) = 0;
  virtual torch::Device device() const = 0;

protected:
  size_t max_size_;
  size_t min_size_;
  torch::Device device_;
};

class UniformReplayBuffer : public ReplayBuffer, public std::enable_shared_from_this<ReplayBuffer> {

public:
  // constructor
  UniformReplayBuffer(size_t max_size, size_t min_size, float gamma, int nstep,
                      RewardReductionMode reward_reduction_mode, int device)
      : ReplayBuffer(max_size, min_size, device), rng_(), gamma_(gamma), nstep_(nstep) {

    // set up reward reduction mode
    skip_incomplete_steps_ = true;
    if (reward_reduction_mode == RewardReductionMode::MeanNoSkip) {
      reward_reduction_mode_ = RewardReductionMode::Mean;
      skip_incomplete_steps_ = false;
    } else if (reward_reduction_mode == RewardReductionMode::WeightedMeanNoSkip) {
      reward_reduction_mode_ = RewardReductionMode::WeightedMean;
      skip_incomplete_steps_ = false;
    } else if (reward_reduction_mode == RewardReductionMode::SumNoSkip) {
      reward_reduction_mode_ = RewardReductionMode::Sum;
      skip_incomplete_steps_ = false;
    } else {
      reward_reduction_mode_ = reward_reduction_mode;
    }
  }

  // disable copy constructor
  UniformReplayBuffer(const UniformReplayBuffer&) = delete;

  // update
  void update(torch::Tensor s, torch::Tensor a, torch::Tensor sp, float r, bool d) {

    // add no grad guard
    torch::NoGradGuard no_grad;

    // clone the tensors and move to device
    auto sc = s.to(device_, s.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto ac = a.to(device_, a.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto spc = sp.to(device_, sp.dtype(), /* non_blocking = */ false, /* copy = */ true);

    // add the newest element to the back
    buffer_.push_back(std::make_tuple(sc, ac, spc, r, d));

    // if we reached max size already, remove the oldest element
    if (buffer_.size() > max_size_) {
      buffer_.pop_front();
    }
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(int batch_size) {

    // add no grad guard
    torch::NoGradGuard no_grad;

    // we need those
    auto stens_list = std::vector<torch::Tensor>(batch_size);
    auto atens_list = std::vector<torch::Tensor>(batch_size);
    auto sptens_list = std::vector<torch::Tensor>(batch_size);
    auto r_list = std::vector<float>(batch_size);
    auto d_list = std::vector<float>(batch_size);

    // be careful, the interval is CLOSED! We need to exclude the upper bound
    std::uniform_int_distribution<size_t> uniform_dist(0, buffer_.size() - nstep_);
    int sample = 0;
    while (sample < batch_size) {

      // get index
      auto index = uniform_dist(rng_);

      // emit the sample at index
      float r;
      float r_norm = 1.;
      int r_count = 1;
      bool d;
      std::tie(stens_list[sample], atens_list[sample], sptens_list[sample], r_list[sample], d) = buffer_.at(index);
      if (d) {
        d_list[sample] = 1.;
      } else {
        d_list[sample] = 0.;
      }

      // if nstep > 1, perform rollout
      bool skip = false;
      float deff = 1. - d_list[sample];
      for (int off = 1; off < nstep_; ++off) {
        torch::Tensor stmp, atmp;
        std::tie(stmp, atmp, sptens_list[sample], r, d) = buffer_.at(index + off);
        auto gamma_eff = static_cast<float>(std::pow(gamma_, off));
        r_list[sample] += gamma_eff * r;
        r_norm += gamma_eff;
        r_count++;
        if (d) {
          // 1-d = 0
          deff *= 0.;
          if ((off != nstep_ - 1) && skip_incomplete_steps_) {
            skip = true;
          }
        } else {
          // 1-d = 1
          deff *= 1.;
        }
      }
      d_list[sample] = 1. - deff;

      if (skip) {
        continue;
      }

      // reward normalization if requested:
      // mean mode is useful for infinite episodes
      // where there is no final reward
      switch (reward_reduction_mode_) {
      case RewardReductionMode::Mean:
        r_list[sample] /= static_cast<float>(r_count);
        break;
      case RewardReductionMode::WeightedMean:
        r_list[sample] /= r_norm;
        break;
      }

      // increase sample index
      sample++;
    }

    // stack the lists
    auto stens = torch::stack(stens_list, 0).clone();
    auto atens = torch::stack(atens_list, 0).clone();
    auto sptens = torch::stack(sptens_list, 0).clone();

    // create new tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto rtens = torch::from_blob(r_list.data(), {batch_size, 1}, options).clone();
    auto dtens = torch::from_blob(d_list.data(), {batch_size, 1}, options).clone();

    return std::make_tuple(stens, atens, sptens, rtens, dtens);
  }

  BufferEntry get(int index) {

    // sanity checks
    if ((index < 0) || (index >= buffer_.size())) {
      throw std::runtime_error("UniformReplayBuffer::get: index " + std::to_string(index) + " out of bounds [0, " +
                               std::to_string(buffer_.size()) + ").");
    }

    // add no grad guard
    torch::NoGradGuard no_grad;

    // emit the sample at index
    torch::Tensor stens, atens, sptens, rtens, dtens;
    float r;
    bool d;
    std::tie(stens, atens, sptens, r, d) = buffer_.at(index);

    return std::make_tuple(stens, atens, sptens, r, d);
  }

  // check functions
  bool isReady() const { return (buffer_.size() >= min_size_); }

  void reset() {
    buffer_.clear();

    return;
  }

  size_t getSize() const { return buffer_.size(); }

  // save and restore
  void save(const std::string& fname) const {
    // create an ordered dict with the buffer contents:
    std::vector<torch::Tensor> s_data, a_data, sp_data;
    std::vector<torch::Tensor> r_data, d_data;
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto options_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    for (size_t index = 0; index < buffer_.size(); ++index) {
      torch::Tensor s, a, sp;
      float r;
      bool d;
      std::tie(s, a, sp, r, d) = buffer_.at(index);
      s_data.push_back(s.to(torch::kCPU));
      a_data.push_back(a.to(torch::kCPU));
      sp_data.push_back(sp.to(torch::kCPU));

      auto rt = torch::from_blob(&r, {1}, options_f).clone();
      r_data.push_back(rt);
      auto dt = torch::from_blob(&d, {1}, options_b).clone();
      d_data.push_back(dt);
    }

    // create subdirectory:
    using namespace torchfort;
    std::filesystem::path root_dir(fname);

    if (!std::filesystem::exists(root_dir)) {
      bool rv = std::filesystem::create_directory(root_dir);
      if (!rv) {
        throw std::runtime_error("UniformReplayBuffer::save: uable to create directory " + root_dir.native() + ".");
      }
    }

    // save the buffer:
    torch::save(s_data, root_dir / "s_data.pt");
    torch::save(sp_data, root_dir / "sp_data.pt");
    torch::save(a_data, root_dir / "a_data.pt");
    torch::save(r_data, root_dir / "r_data.pt");
    torch::save(d_data, root_dir / "d_data.pt");

    return;
  }

  void load(const std::string& fname) {
    // get vectors for buffers:
    std::vector<torch::Tensor> s_data, a_data, sp_data;
    std::vector<torch::Tensor> r_data, d_data;

    using namespace torchfort;
    std::filesystem::path root_dir(fname);

    torch::load(s_data, root_dir / "s_data.pt");
    torch::load(a_data, root_dir / "a_data.pt");
    torch::load(sp_data, root_dir / "sp_data.pt");
    torch::load(r_data, root_dir / "r_data.pt");
    torch::load(d_data, root_dir / "d_data.pt");

    // iterate over loaded data and populate buffer
    buffer_.clear();
    for (size_t index = 0; index < s_data.size(); ++index) {
      auto s = s_data[index];
      auto a = a_data[index];
      auto sp = sp_data[index];
      float r = r_data[index].item<float>();
      bool d = d_data[index].item<bool>();

      // update buffer
      this->update(s, a, sp, r, d);
    }
  }

  void printInfo() const {
    std::cout << "uniform replay buffer parameters:" << std::endl;
    std::cout << "max_size = " << max_size_ << std::endl;
    std::cout << "min_size = " << min_size_ << std::endl;
  }

  torch::Device device() const { return device_; }

private:
  // the rbuffer contains tuples: (s, a, s', r, d)
  // std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, bool>> buffer_;
  std::deque<BufferEntry> buffer_;
  // rng
  std::mt19937_64 rng_;
  // some parameters:
  float gamma_;
  int nstep_;
  RewardReductionMode reward_reduction_mode_;
  bool skip_incomplete_steps_;
};

} // namespace rl
} // namespace torchfort
