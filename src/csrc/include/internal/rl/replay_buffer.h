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
using SampleResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                torch::Tensor, torch::Tensor>;

enum RewardReductionMode { Sum = 1, Mean = 2, WeightedMean = 3, SumNoSkip = 4, MeanNoSkip = 5, WeightedMeanNoSkip = 6 };

// abstract base class for replay buffer
class ReplayBuffer {
public:
  // disable copy constructor
  ReplayBuffer(const ReplayBuffer&) = delete;
  // base constructor
  ReplayBuffer(size_t max_size, size_t min_size, size_t n_envs, int device)
      : max_size_(max_size / n_envs), min_size_(min_size / n_envs), n_envs_(n_envs), device_(get_device(device)) {
    // some asserts
    if ((max_size < n_envs) || (min_size < n_envs)) {
      throw std::runtime_error("ReplayBuffer::ReplayBuffer: Error, make sure the buffer min and max buffer sizes are "
                               "bigger than or equal to the number of environments");
    }
  }

  // accessor functions
  size_t getMinSize() const { return min_size_ * n_envs_; }
  size_t getMaxSize() const { return max_size_ * n_envs_; }
  size_t nEnvs() const { return n_envs_; }

  // virtual functions
  virtual void update(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) = 0;
  // sample elements: returns (s, a, s', r, d, is_weights, indices)
  // is_weights are all-ones and indices are all-zeros for uniform sampling
  virtual SampleResult sample(int) = 0;
  // update priorities based on per-sample TD errors; no-op for uniform sampling
  virtual void update_priorities(torch::Tensor /* indices */, torch::Tensor /* td_errors */) {}
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

    if ((s.sizes()[0] != n_envs_) || (a.sizes()[0] != n_envs_) || (sp.sizes()[0] != n_envs_)) {
      throw std::runtime_error("UniformReplayBuffer::update: the size of the leading dimension of tensors s, a and sp "
                               "has to be equal to the number of environments");
    }
    if ((r.sizes()[0] != n_envs_) || (d.sizes()[0] != n_envs_)) {
      throw std::runtime_error("UniformReplayBuffer::update: tensors r and d have to be one dimensional and the size "
                               "has to be equal to the number of environments");
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

  SampleResult sample(int batch_size) {

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

    // uniform buffer: unit IS weights, zero indices (ignored by no-op update_priorities)
    auto is_weights = torch::ones(batch_size, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto indices = torch::zeros(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device_));

    return std::make_tuple(stens, atens, sptens, rtens, dtens, is_weights, indices);
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

  void setSeed(unsigned int seed) { rng_.seed(seed); }

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

// Prioritized Experience Replay buffer (Schaul et al., 2015).
//
// Priorities are stored in a sum-tree so that O(log N) sampling and updates
// are possible.  Each physical buffer slot holds one timestep for all n_envs_
// environments.  Priorities operate at the step level; the env is chosen
// uniformly given the step.
//
// IS weight formula (normalised to [0,1]):
//   w_i = (min_p_alpha / p_i)^beta
// where p_i = p_raw^alpha is the value stored in the tree leaf and
// min_p_alpha is the minimum non-zero p^alpha currently tracked in the buffer.
// max weight = 1 by construction (achieved by the lowest-priority entry).
//
// When a new entry is written at position write_pos_:
//   1. Set its tree priority to 0 (not yet a valid n-step starting point).
//   2. Enable position (write_pos_ - nstep_ + 1) % max_size_ with max_priority
//      (it now has nstep_ consecutive entries ahead of it).
// This keeps the circular-buffer n-step invariant correct without any
// extra bookkeeping.
class PrioritizedReplayBuffer : public ReplayBuffer, public std::enable_shared_from_this<ReplayBuffer> {

public:
  PrioritizedReplayBuffer(size_t max_size, size_t min_size, size_t n_envs, float gamma, int nstep,
                          RewardReductionMode reward_reduction_mode, float alpha, float beta0, float beta_max,
                          size_t beta_steps, int device)
      : ReplayBuffer(max_size, min_size, n_envs, device), gamma_(gamma), nstep_(nstep), alpha_(alpha), beta_(beta0),
        beta_max_(beta_max),
        beta_increment_(beta_steps > 0 ? (beta_max - beta0) / static_cast<float>(beta_steps) : 0.f),
        epsilon_(1e-6f), max_priority_(1.f), min_p_alpha_(std::numeric_limits<float>::max()),
        write_pos_(0), current_size_(0) {

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

    buffer_.resize(max_size_);
    // 1-indexed sum-tree: root at 1, leaves at [max_size_, 2*max_size_ - 1]
    sum_tree_.assign(2 * max_size_, 0.f);
    priorities_.assign(max_size_, 0.f);
  }

  PrioritizedReplayBuffer(const PrioritizedReplayBuffer&) = delete;

  void update(torch::Tensor s, torch::Tensor a, torch::Tensor sp, torch::Tensor r, torch::Tensor d) override {
    torch::NoGradGuard no_grad;

    if ((s.sizes()[0] != n_envs_) || (a.sizes()[0] != n_envs_) || (sp.sizes()[0] != n_envs_)) {
      throw std::runtime_error(
          "PrioritizedReplayBuffer::update: leading dimension of s, a, sp must equal n_envs");
    }
    if ((r.sizes()[0] != n_envs_) || (d.sizes()[0] != n_envs_)) {
      throw std::runtime_error(
          "PrioritizedReplayBuffer::update: leading dimension of r, d must equal n_envs");
    }

    auto sc  = s.to(device_,  s.dtype(),  false, true);
    auto ac  = a.to(device_,  a.dtype(),  false, true);
    auto spc = sp.to(device_, sp.dtype(), false, true);
    auto rc  = r.to(device_,  r.dtype(),  false, true);
    auto dc  = d.to(device_,  d.dtype(),  false, true);

    // write entry; zero priority until it becomes a valid n-step starting point
    size_t pos = write_pos_;
    buffer_[pos] = std::make_tuple(sc, ac, spc, rc, dc);
    treeUpdate_(pos, 0.f);
    priorities_[pos] = 0.f;

    write_pos_ = (write_pos_ + 1) % max_size_;
    current_size_ = std::min(current_size_ + 1, max_size_);

    // the position that can now start a complete n-step rollout ending at pos
    if (current_size_ >= static_cast<size_t>(nstep_)) {
      size_t valid_pos = (pos + max_size_ - nstep_ + 1) % max_size_;
      float p_alpha = std::pow(max_priority_, alpha_);
      treeUpdate_(valid_pos, p_alpha);
      priorities_[valid_pos] = max_priority_;
      min_p_alpha_ = std::min(min_p_alpha_, p_alpha);
    }
  }

  SampleResult sample(int batch_size) override {
    torch::NoGradGuard no_grad;

    auto stens_list  = std::vector<torch::Tensor>(batch_size);
    auto atens_list  = std::vector<torch::Tensor>(batch_size);
    auto sptens_list = std::vector<torch::Tensor>(batch_size);
    auto rtens_list  = std::vector<torch::Tensor>(batch_size);
    auto dtens_list  = std::vector<torch::Tensor>(batch_size);
    auto wtens_list  = std::vector<torch::Tensor>(batch_size);
    auto itens_list  = std::vector<torch::Tensor>(batch_size);

    // anneal beta towards beta_max_
    beta_ = std::min(beta_max_, beta_ + beta_increment_);

    float total    = treeTotal_();
    float segment  = total / static_cast<float>(batch_size);
    // minimum p^alpha currently in the buffer — normalises IS weights so max weight = 1
    float min_p_alpha = min_p_alpha_;

    std::uniform_int_distribution<size_t> env_dist(0, n_envs_ - 1);

    int s = 0;
    while (s < batch_size) {
      // stratified sampling: one draw per equal-width segment of the priority sum
      float lo = segment * static_cast<float>(s);
      float hi = segment * static_cast<float>(s + 1);
      std::uniform_real_distribution<float> seg_dist(lo, hi);
      size_t pos = treeSample_(seg_dist(rng_));

      // guard against numerical edge-cases (leaf with zero priority)
      float p_i = sum_tree_[max_size_ + pos];
      if (p_i <= 0.f) {
        continue;
      }

      int64_t env_idx = static_cast<int64_t>(env_dist(rng_));

      // IS weight: w_i = (min_p_alpha / p_i)^beta  — max weight = 1 by construction
      float weight = std::pow(min_p_alpha / p_i, beta_);

      // extract (state, action, ...) at (pos, env_idx)
      torch::Tensor stens, atens, sptens, rtens, dtens;
      std::tie(stens, atens, sptens, rtens, dtens) = buffer_[pos];
      stens_list[s]  = stens.index({env_idx, "..."}).clone();
      atens_list[s]  = atens.index({env_idx, "..."}).clone();
      sptens_list[s] = sptens.index({env_idx, "..."}).clone();
      rtens_list[s]  = rtens.index({env_idx}).clone();
      dtens_list[s]  = dtens.index({env_idx}).clone();

      // n-step rollout — identical logic to UniformReplayBuffer, but uses modular indexing
      float r_norm = 1.f;
      int   r_count = 1;
      bool  skip = false;
      torch::Tensor deff = 1.f - dtens_list[s];
      for (int off = 1; off < nstep_; ++off) {
        size_t next_pos = (pos + static_cast<size_t>(off)) % max_size_;
        std::tie(stens, atens, sptens, rtens, dtens) = buffer_[next_pos];
        sptens_list[s] = sptens.index({env_idx, "..."}).clone();
        float gamma_eff = static_cast<float>(std::pow(gamma_, off));
        rtens_list[s]  = rtens_list[s] + gamma_eff * rtens.index({env_idx});
        r_norm  += gamma_eff;
        r_count++;

        float d_val = dtens.index({env_idx}).item<float>();
        if (std::abs(d_val - 1.f) < 1e-6f) {
          deff = deff * 0.f;
          if ((off != nstep_ - 1) && skip_incomplete_steps_) {
            skip = true;
          }
        }
      }
      dtens_list[s] = (1.f - deff).clone();

      if (skip) {
        continue; // retry slot s without incrementing
      }

      switch (reward_reduction_mode_) {
      case RewardReductionMode::Mean:
        rtens_list[s] = rtens_list[s] / static_cast<float>(r_count);
        break;
      case RewardReductionMode::WeightedMean:
        rtens_list[s] = rtens_list[s] / r_norm;
        break;
      default:
        break;
      }

      auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
      auto long_opts  = torch::TensorOptions().dtype(torch::kLong).device(device_);
      wtens_list[s] = torch::tensor(weight, float_opts);
      itens_list[s] = torch::tensor(static_cast<int64_t>(pos), long_opts);

      ++s;
    }

    return std::make_tuple(torch::stack(stens_list, 0).clone(), torch::stack(atens_list, 0).clone(),
                           torch::stack(sptens_list, 0).clone(), torch::stack(rtens_list, 0).clone(),
                           torch::stack(dtens_list, 0).clone(), torch::stack(wtens_list, 0).clone(),
                           torch::stack(itens_list, 0).clone());
  }

  void update_priorities(torch::Tensor indices, torch::Tensor td_errors) override {
    torch::NoGradGuard no_grad;

    auto indices_cpu = indices.to(torch::kCPU).contiguous();
    auto td_errors_cpu = td_errors.to(torch::kCPU).contiguous();
    auto idx_acc = indices_cpu.accessor<int64_t, 1>();
    auto td_acc  = td_errors_cpu.accessor<float, 1>();

    for (int64_t i = 0; i < idx_acc.size(0); ++i) {
      size_t pos    = static_cast<size_t>(idx_acc[i]);
      float raw_p   = std::abs(td_acc[i]) + epsilon_;
      float p_alpha = std::pow(raw_p, alpha_);
      treeUpdate_(pos, p_alpha);
      priorities_[pos] = raw_p;
      max_priority_ = std::max(max_priority_, raw_p);
      min_p_alpha_  = std::min(min_p_alpha_,  p_alpha);
    }
  }

  BufferEntry get(int index) override {
    if (index < 0 || static_cast<size_t>(index) >= current_size_) {
      throw std::runtime_error("PrioritizedReplayBuffer::get: index " + std::to_string(index) +
                               " out of bounds [0, " + std::to_string(current_size_) + ").");
    }
    return buffer_[index];
  }

  bool isReady() const override { return current_size_ >= min_size_; }

  void reset() override {
    current_size_ = 0;
    write_pos_    = 0;
    max_priority_ = 1.f;
    min_p_alpha_  = std::numeric_limits<float>::max();
    std::fill(sum_tree_.begin(),   sum_tree_.end(),   0.f);
    std::fill(priorities_.begin(), priorities_.end(), 0.f);
  }

  size_t getSize() const override { return current_size_; }

  void setSeed(unsigned int seed) override { rng_.seed(seed); }

  torch::Device device() const override { return device_; }

  void printInfo() const override {
    std::cout << "prioritized replay buffer parameters:" << std::endl;
    std::cout << "max_size = " << max_size_ << std::endl;
    std::cout << "min_size = " << min_size_ << std::endl;
    std::cout << "n_envs = " << n_envs_ << std::endl;
    std::cout << "alpha = " << alpha_ << std::endl;
    std::cout << "beta (current) = " << beta_ << std::endl;
    std::cout << "beta_max = " << beta_max_ << std::endl;
    std::cout << "epsilon = " << epsilon_ << std::endl;
  }

  void save(const std::string& fname) const override {
    std::vector<torch::Tensor> s_data, a_data, sp_data, r_data, d_data;
    for (size_t i = 0; i < current_size_; ++i) {
      torch::Tensor s, a, sp, r, d;
      std::tie(s, a, sp, r, d) = buffer_[i];
      s_data.push_back(s.to(torch::kCPU));
      a_data.push_back(a.to(torch::kCPU));
      sp_data.push_back(sp.to(torch::kCPU));
      r_data.push_back(r.to(torch::kCPU));
      d_data.push_back(d.to(torch::kCPU));
    }

    std::filesystem::path root_dir(fname);
    if (!std::filesystem::exists(root_dir)) {
      bool rv = std::filesystem::create_directory(root_dir);
      if (!rv) {
        throw std::runtime_error("PrioritizedReplayBuffer::save: unable to create directory " +
                                 root_dir.native() + ".");
      }
    }

    torch::save(s_data,  root_dir / "s_data.pt");
    torch::save(a_data,  root_dir / "a_data.pt");
    torch::save(sp_data, root_dir / "sp_data.pt");
    torch::save(r_data,  root_dir / "r_data.pt");
    torch::save(d_data,  root_dir / "d_data.pt");

    // save raw priorities (before alpha) for positions 0..current_size_-1
    auto prio_tensor = torch::tensor(
        std::vector<float>(priorities_.begin(), priorities_.begin() + current_size_));
    torch::save({prio_tensor}, root_dir / "priorities.pt");

    // save scalar state: [write_pos, current_size, beta, max_priority, min_p_alpha]
    auto state_tensor = torch::tensor(std::vector<float>{static_cast<float>(write_pos_),
                                                          static_cast<float>(current_size_), beta_,
                                                          max_priority_, min_p_alpha_});
    torch::save({state_tensor}, root_dir / "per_state.pt");
  }

  void load(const std::string& fname) override {
    std::filesystem::path root_dir(fname);

    std::vector<torch::Tensor> s_data, a_data, sp_data, r_data, d_data;
    torch::load(s_data,  root_dir / "s_data.pt");
    torch::load(a_data,  root_dir / "a_data.pt");
    torch::load(sp_data, root_dir / "sp_data.pt");
    torch::load(r_data,  root_dir / "r_data.pt");
    torch::load(d_data,  root_dir / "d_data.pt");

    // restore scalar state
    std::vector<torch::Tensor> state_vec;
    torch::load(state_vec, root_dir / "per_state.pt");
    write_pos_    = static_cast<size_t>(state_vec[0][0].item<float>());
    current_size_ = static_cast<size_t>(state_vec[0][1].item<float>());
    beta_         = state_vec[0][2].item<float>();
    max_priority_ = state_vec[0][3].item<float>();
    min_p_alpha_  = state_vec[0][4].item<float>();

    // restore buffer entries at their physical positions
    buffer_.assign(max_size_, BufferEntry{});
    for (size_t i = 0; i < s_data.size(); ++i) {
      buffer_[i] = std::make_tuple(s_data[i], a_data[i], sp_data[i], r_data[i], d_data[i]);
    }

    // restore priorities and rebuild sum-tree
    std::vector<torch::Tensor> prio_vec;
    torch::load(prio_vec, root_dir / "priorities.pt");
    std::fill(priorities_.begin(), priorities_.end(), 0.f);
    std::fill(sum_tree_.begin(),   sum_tree_.end(),   0.f);
    for (size_t i = 0; i < current_size_; ++i) {
      float raw_p = prio_vec[0][static_cast<int64_t>(i)].item<float>();
      priorities_[i] = raw_p;
      if (raw_p > 0.f) {
        treeUpdate_(i, std::pow(raw_p, alpha_));
      }
    }
  }

private:
  // update one leaf and propagate up to root — O(log N)
  void treeUpdate_(size_t pos, float priority_alpha) {
    size_t idx = max_size_ + pos;
    sum_tree_[idx] = priority_alpha;
    while (idx > 1) {
      idx >>= 1;
      sum_tree_[idx] = sum_tree_[2 * idx] + sum_tree_[2 * idx + 1];
    }
  }

  // traverse the tree to find the leaf whose prefix-sum contains value — O(log N)
  size_t treeSample_(float value) const {
    // clamp to avoid floating-point overshoot past the last leaf
    value = std::min(value, treeTotal_() * (1.f - 1e-6f));
    size_t idx = 1;
    while (idx < max_size_) {
      size_t left = 2 * idx;
      if (value <= sum_tree_[left]) {
        idx = left;
      } else {
        value -= sum_tree_[left];
        idx = left + 1;
      }
    }
    return idx - max_size_;
  }

  float treeTotal_() const { return sum_tree_[1]; }

  // circular buffer (vector for O(1) indexed access)
  std::vector<BufferEntry> buffer_;
  // 1-indexed sum-tree of size 2*max_size_; leaf for position p at index max_size_+p
  std::vector<float> sum_tree_;
  // raw priorities (before alpha exponent), same indexing as buffer_
  std::vector<float> priorities_;

  size_t write_pos_;
  size_t current_size_;

  float alpha_;           // priority exponent
  float beta_;            // IS weight exponent (annealed from beta0 toward beta_max_)
  float beta_max_;
  float beta_increment_;  // added to beta_ each call to sample()
  float epsilon_;         // priority floor (prevents zero probability)
  float max_priority_;    // maximum raw priority seen; assigned to new entries
  float min_p_alpha_;     // minimum p^alpha currently tracked; used to normalise IS weights

  std::mt19937_64 rng_;
  float gamma_;
  int nstep_;
  RewardReductionMode reward_reduction_mode_;
  bool skip_incomplete_steps_;
};

} // namespace rl
} // namespace torchfort
