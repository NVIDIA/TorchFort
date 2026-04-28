/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>

#include <torch/torch.h>

#include "internal/distributed.h"

namespace torchfort {

namespace rl {

// Online per-feature normalizer using Welford's parallel algorithm.
//
// Running statistics (mean, M2, count) are stored on CPU. normalize() moves them
// to the input tensor's device on-the-fly so the normalization arithmetic runs on
// GPU when called with device tensors.
//
// Two normalization modes are supported via the scale_only constructor flag:
//
//   scale_only = false (default): x_norm = (x - mean) / sqrt(var + eps)
//     Use for observations/states where zero-centering is desirable.
//
//   scale_only = true:            x_norm = x / sqrt(var + eps)
//     Use for returns, where the mean must be preserved so the value function
//     can learn the correct absolute level. The mean is still tracked internally
//     (for distributed sync via Chan's algorithm) but not subtracted during normalization.
//
// Distributed sync: call sync() once per training step to combine per-rank running
// statistics across MPI ranks using Chan's parallel algorithm via two allreduce calls:
//   1. allreduce(count, weighted_mean)  -> global count and mean
//   2. allreduce(local M2 contribution) -> global M2
class RunningNormalizer {
public:
  explicit RunningNormalizer(float eps = 1e-8f, bool scale_only = false)
      : count_(0), eps_(eps), scale_only_(scale_only) {}

  // Update running statistics with a batch of samples.
  // x shape: [batch, feature...]. Statistics are tracked per feature element.
  // x may be on any device; statistics are always kept on CPU.
  void update(torch::Tensor x);

  // Normalize x using current running statistics.
  // Returns x unchanged if fewer than 2 samples have been seen.
  // Statistics are moved to x.device() for the computation.
  // In scale_only mode, only divides by std without subtracting the mean.
  torch::Tensor normalize(torch::Tensor x) const;

  // Combine running statistics across MPI ranks using Chan's parallel algorithm.
  // No-op if comm is null or count_ == 0.
  void sync(std::shared_ptr<Comm> comm);

  // Checkpoint support.
  void save(const std::string& path) const;
  void load(const std::string& path);

  bool isInitialized() const { return count_ > 0; }

private:
  torch::Tensor mean_; // per-feature mean, CPU float32
  torch::Tensor M2_;   // per-feature sum of squared deviations, CPU float32
  int64_t count_;
  float eps_;
  bool scale_only_;
};

} // namespace rl

} // namespace torchfort
