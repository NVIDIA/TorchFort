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

  using BufferEntry = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

enum RewardReductionMode { Sum = 1, Mean = 2, WeightedMean = 3, SumNoSkip = 4, MeanNoSkip = 5, WeightedMeanNoSkip = 6 };

// abstract base class for replay buffer
class ReplayBuffer {
public:
  // disable copy constructor
  ReplayBuffer(const ReplayBuffer&) = delete;
  // base constructor
  ReplayBuffer(size_t max_size, size_t min_size, size_t n_envs, int device)
    : max_size_(max_size/n_envs), min_size_(min_size/n_envs), n_envs_(n_envs), device_(get_device(device)) {
    // some asserts
    if ((max_size < n_envs) || (min_size < n_envs)) {
      throw std::runtime_error("ReplayBuffer::ReplayBuffer: Error, make sure the buffer min and max buffer sizes are bigger than or equal to the number of environments");
    }
  }

  // accessor functions
  size_t getMinSize() const { return min_size_ * n_envs_; }
  size_t getMaxSize() const { return max_size_ * n_envs_; }
  size_t nEnvs() const { return n_envs_; }

  // virtual functions
  virtual void update(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) = 0;
  // sample element randomly
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(int) = 0;
  // get specific element
  virtual BufferEntry get(int) = 0;
  // helper functions
  virtual bool isReady() const = 0;
  virtual void reset() = 0;
  virtual size_t getSize() const = 0;
  virtual void setSeed(unsigned int seed) = 0;
  virtual void printInfo() const = 0;
  virtual void save(const std::string& fname) const = 0;
  virtual void load(const std::string& fname) = 0;
  virtual torch::Device device() const = 0;

protected:
  size_t max_size_;
  size_t min_size_;
  size_t n_envs_;
  torch::Device device_;
};

class UniformReplayBuffer : public ReplayBuffer, public std::enable_shared_from_this<ReplayBuffer> {

public:
  // constructor
  UniformReplayBuffer(size_t max_size, size_t min_size, size_t n_envs, float gamma, int nstep,
                      RewardReductionMode reward_reduction_mode, int device)
    : ReplayBuffer(max_size, min_size, n_envs, device), rng_(), gamma_(gamma), nstep_(nstep) {

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
  void update(torch::Tensor s, torch::Tensor a, torch::Tensor sp, torch::Tensor r, torch::Tensor d) {

    // add no grad guard
    torch::NoGradGuard no_grad;

    if( (s.sizes()[0] != n_envs_) || (a.sizes()[0] != n_envs_) || (sp.sizes()[0] != n_envs_) ) {
      throw std::runtime_error("UniformReplayBuffer::update: the size of the leading dimension of tensors s, a and sp has to be equal to the number of environments");
    }
    if ( (r.sizes()[0] != n_envs_) || (d.sizes()[0] != n_envs_) ) {
      throw std::runtime_error("UniformReplayBuffer::update: tensors r and d have to be one dimensional and the size has to be equal to the number of environments");
    }

    // clone the tensors and move to device
    auto sc = s.to(device_, s.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto ac = a.to(device_, a.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto spc = sp.to(device_, sp.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto rc = r.to(device_, r.dtype(), /* non_blocking = */ false, /* copy = */ true);
    auto dc = d.to(device_, d.dtype(), /* non_blocking = */ false, /* copy = */ true);

    // add the newest element to the back
    buffer_.push_back(std::make_tuple(sc, ac, spc, rc, dc));

    // if we reached max size already, remove the oldest element
    if (buffer_.size() > max_size_) {
      buffer_.pop_front();
    }
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(int batch_size) {

    // add no grad guard
    torch::NoGradGuard no_grad;

    // we need those
    torch::Tensor stens, atens, sptens, rtens, dtens;
    auto stens_list = std::vector<torch::Tensor>(batch_size);
    auto atens_list = std::vector<torch::Tensor>(batch_size);
    auto sptens_list = std::vector<torch::Tensor>(batch_size);
    auto rtens_list = std::vector<torch::Tensor>(batch_size);
    auto dtens_list = std::vector<torch::Tensor>(batch_size);

    // we need the tensor options too
    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    // be careful, the interval is CLOSED! We need to exclude the upper bound
    std::uniform_int_distribution<size_t> uniform_dist(0, (buffer_.size() - nstep_) * n_envs_);
    int sample = 0;
    while (sample < batch_size) {

      // global index
      auto glob_idx = uniform_dist(rng_);
      
      // we assume that glob_idx = env_idx + n_envs * step_idx
      int64_t env_idx = glob_idx % n_envs_;
      int64_t step_idx = glob_idx / n_envs_;
      
      // emit the sample at index
      std::tie(stens, atens, sptens, rtens, dtens) = buffer_.at(step_idx);

      // extract values at env index
      stens_list[sample] = stens.index({env_idx, "..."}).clone();
      atens_list[sample] = atens.index({env_idx, "..."}).clone();
      sptens_list[sample] = sptens.index({env_idx, "..."}).clone();
      rtens_list[sample] = rtens.index({env_idx}).clone();
      dtens_list[sample] = dtens.index({env_idx}).clone();
      
      // if nstep > 1, perform rollout
      float r_norm = 1.;
      int r_count = 1;
      bool skip = false;
      torch::Tensor deff = 1. - dtens_list[sample];
      for (int off = 1; off < nstep_; ++off) {
        std::tie(stens, atens, sptens, rtens, dtens) = buffer_.at(step_idx + off);
	sptens_list[sample] = sptens.index({env_idx, "..."}).clone();
        auto gamma_eff = static_cast<float>(std::pow(gamma_, off));
        rtens_list[sample] += gamma_eff * rtens.index({env_idx});
        r_norm += gamma_eff;
        r_count++;

	// update deff
	float d = dtens.index({env_idx}).item<float>();
        if (std::abs(d - 1.) < 1.e-6) {
          // episode ended: 1-d = 0
          deff *= 0.;
          if ((off != nstep_ - 1) && skip_incomplete_steps_) {
            skip = true;
          }
        } else {
          // episode continues: 1-d = 1
          deff *= 1.;
        }
      }
      dtens_list[sample] = (1. - deff).clone();

      if (skip) {
        continue;
      }

      // reward normalization if requested:
      // mean mode is useful for infinite episodes
      // where there is no final reward
      switch (reward_reduction_mode_) {
      case RewardReductionMode::Mean:
        rtens_list[sample] /= static_cast<float>(r_count);
        break;
      case RewardReductionMode::WeightedMean:
        rtens_list[sample] /= r_norm;
        break;
      }

      // increase sample index
      sample++;
    }

    // stack the lists
    stens = torch::stack(stens_list, 0).clone();
    atens = torch::stack(atens_list, 0).clone();
    sptens = torch::stack(sptens_list, 0).clone();
    rtens = torch::stack(rtens_list, 0).clone();
    dtens = torch::stack(dtens_list, 0).clone();

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
    std::tie(stens, atens, sptens, rtens, dtens) = buffer_.at(index);

    return std::make_tuple(stens, atens, sptens, rtens, dtens);
  }

  // check functions
  bool isReady() const { return (buffer_.size() >= min_size_); }

  void reset() {
    buffer_.clear();

    return;
  }

  size_t getSize() const { return buffer_.size(); }

  void setSeed(unsigned int seed) {
    rng_.seed(seed);
  }

  // save and restore
  void save(const std::string& fname) const {
    // create an ordered dict with the buffer contents:
    std::vector<torch::Tensor> s_data, a_data, sp_data, r_data, d_data;
    for (size_t index = 0; index < buffer_.size(); ++index) {
      torch::Tensor s, a, sp, r, d;
      std::tie(s, a, sp, r, d) = buffer_.at(index);
      s_data.push_back(s.to(torch::kCPU));
      a_data.push_back(a.to(torch::kCPU));
      sp_data.push_back(sp.to(torch::kCPU));
      r_data.push_back(r.to(torch::kCPU));
      d_data.push_back(d.to(torch::kCPU));
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
    std::vector<torch::Tensor> s_data, a_data, sp_data, r_data, d_data;

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
      auto r = r_data[index];
      auto d = d_data[index];

      // update buffer
      this->update(s, a, sp, r, d);
    }
  }

  void printInfo() const {
    std::cout << "uniform replay buffer parameters:" << std::endl;
    std::cout << "max_size = " << max_size_ << std::endl;
    std::cout << "min_size = " << min_size_ << std::endl;
    std::cout << "n_envs = " << n_envs_ << std::endl;
  }

  torch::Device device() const { return device_; }

private:
  // the rbuffer contains tuples: (s, a, s', r, d)
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
