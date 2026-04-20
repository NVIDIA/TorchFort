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

#include "internal/rl/running_normalizer.h"

namespace torchfort {

namespace rl {

void RunningNormalizer::update(torch::Tensor x) {
  torch::NoGradGuard no_grad;

  // move to CPU float32, flatten to [batch, features]
  int64_t batch_size = x.size(0);
  auto x_flat = x.reshape({batch_size, -1}).to(torch::kFloat32).cpu();

  // batch statistics
  auto batch_mean = x_flat.mean(0);
  auto batch_M2 = torch::sum(torch::square(x_flat - batch_mean.unsqueeze(0)), 0);

  if (count_ == 0) {
    mean_ = batch_mean;
    M2_ = batch_M2;
    count_ = batch_size;
  } else {
    // Chan's parallel algorithm: combine (count_, mean_, M2_) with (batch_size, batch_mean, batch_M2)
    int64_t new_count = count_ + batch_size;
    auto delta = batch_mean - mean_;
    auto new_mean = mean_ + delta * (static_cast<float>(batch_size) / static_cast<float>(new_count));
    auto new_M2 = M2_ + batch_M2 +
                  torch::square(delta) *
                      (static_cast<float>(count_) * static_cast<float>(batch_size) / static_cast<float>(new_count));
    count_ = new_count;
    mean_ = new_mean;
    M2_ = new_M2;
  }
}

torch::Tensor RunningNormalizer::normalize(torch::Tensor x) const {
  if (count_ < 2) return x;

  torch::NoGradGuard no_grad;

  auto orig_shape = x.sizes().vec();
  int64_t batch_size = x.size(0);

  // flatten to [batch, features], normalize, restore shape
  auto x_flat = x.reshape({batch_size, -1}).to(torch::kFloat32);

  auto var = M2_ / static_cast<float>(count_ - 1);
  auto std = torch::sqrt(var + eps_).to(x.device());

  if (scale_only_) {
    // preserve the mean: divide by std only (used for return normalization)
    return (x_flat / std).reshape(orig_shape);
  } else {
    auto mean = mean_.to(x.device());
    return ((x_flat - mean) / std).reshape(orig_shape);
  }
}

void RunningNormalizer::sync(std::shared_ptr<Comm> comm) {
  if (!comm || count_ == 0) return;

  torch::NoGradGuard no_grad;

  // Step 1: compute global count and global mean via allreduce of (count, count*mean).
  // Using false (sum, not average) so we get the global sums directly.
  auto count_tensor = torch::tensor({static_cast<float>(count_)});
  auto weighted_mean = mean_ * static_cast<float>(count_);

  std::vector<torch::Tensor> step1 = {count_tensor, weighted_mean};
  comm->allreduce(step1, false);

  float global_count = step1[0].item<float>();
  auto global_mean = step1[1] / global_count;

  // Step 2: combine M2 across ranks using Chan's formula.
  // Each rank contributes: M2_i + n_i * (mean_i - global_mean)^2
  auto local_contribution = M2_ + static_cast<float>(count_) * torch::square(mean_ - global_mean);
  std::vector<torch::Tensor> step2 = {local_contribution};
  comm->allreduce(step2, false);

  count_ = static_cast<int64_t>(global_count);
  mean_ = global_mean;
  M2_ = step2[0];
}

void RunningNormalizer::save(const std::string& path) const {
  torch::serialize::OutputArchive archive;
  archive.write("mean", mean_.defined() ? mean_ : torch::zeros({1}));
  archive.write("M2", M2_.defined() ? M2_ : torch::zeros({1}));
  archive.write("count", torch::tensor({count_}));
  archive.write("scale_only", torch::tensor({static_cast<int64_t>(scale_only_)}));
  archive.save_to(path);
}

void RunningNormalizer::load(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  archive.read("mean", mean_);
  archive.read("M2", M2_);
  torch::Tensor count_tensor;
  archive.read("count", count_tensor);
  count_ = count_tensor.item<int64_t>();
  torch::Tensor scale_only_tensor;
  archive.read("scale_only", scale_only_tensor);
  scale_only_ = static_cast<bool>(scale_only_tensor.item<int64_t>());
}

} // namespace rl

} // namespace torchfort
