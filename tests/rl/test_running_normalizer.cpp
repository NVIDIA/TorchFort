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

#include "internal/rl/running_normalizer.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace torchfort;

// Ground truth distribution parameters for 4 independent features.
// Each feature has a distinct nonzero mean and non-unit std so that a buggy
// normalizer (e.g. one that ignores the mean or gets variance wrong) is
// reliably caught.
static const int N_FEATURES = 4;
static const std::vector<float> TRUE_MEAN = {3.0f, -2.0f, 0.5f, 10.0f};
static const std::vector<float> TRUE_STD  = {1.5f,  0.5f, 3.0f,  0.2f};

// Helper: build a [batch_size, N_FEATURES] tensor sampled from the ground-truth distribution.
static torch::Tensor make_batch(int batch_size) {
  auto mean = torch::tensor(TRUE_MEAN);
  auto std  = torch::tensor(TRUE_STD);
  return torch::randn({batch_size, N_FEATURES}) * std.unsqueeze(0) + mean.unsqueeze(0);
}

// ---- Test 1: statistics accuracy -----------------------------------------
// Feed N batches from a known distribution, then verify that the normalizer's
// running mean and std converge to the true values within a tight tolerance.
// With 50000 total samples the estimation error should be well below 1%.
TEST(RunningNormalizer, StatsAccuracy) {
  torch::manual_seed(42);
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer normalizer;

  const int batch_size = 100;
  const int n_batches  = 500; // 50000 samples total

  for (int i = 0; i < n_batches; ++i) {
    normalizer.update(make_batch(batch_size));
  }

  ASSERT_TRUE(normalizer.isInitialized());

  // Access internal state via a fresh normalize pass on a zero tensor to
  // extract mean and std indirectly, OR test via normalized output.
  // We instead check the running statistics directly by normalizing a tensor
  // whose value we control and inspecting the result.
  //
  // Strategy: normalize(true_mean_tensor) should yield ~0, and
  //           normalize(true_mean_tensor + true_std_tensor) should yield ~1.
  auto mean_tensor = torch::tensor(TRUE_MEAN).unsqueeze(0); // [1, 4]
  auto std_tensor  = torch::tensor(TRUE_STD).unsqueeze(0);

  auto normalized_mean = normalizer.normalize(mean_tensor);
  auto normalized_mean_plus_std = normalizer.normalize(mean_tensor + std_tensor);

  // normalized(true_mean) should be ~0 for each feature
  for (int f = 0; f < N_FEATURES; ++f) {
    EXPECT_NEAR(normalized_mean[0][f].item<float>(), 0.0f, 0.05f)
        << "Feature " << f << ": normalized mean should be ~0";
  }

  // normalized(true_mean + true_std) should be ~1 for each feature
  for (int f = 0; f < N_FEATURES; ++f) {
    EXPECT_NEAR(normalized_mean_plus_std[0][f].item<float>(), 1.0f, 0.05f)
        << "Feature " << f << ": normalized(mean + std) should be ~1";
  }
}

// ---- Test 2: normalized output has zero mean and unit variance -----------
// After training the normalizer, normalize a large fresh batch drawn from
// the same distribution and verify the output is approximately N(0,1).
TEST(RunningNormalizer, NormalizedOutputDistribution) {
  torch::manual_seed(123);
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer normalizer;

  // Warm up the normalizer with 50000 samples
  const int warmup_batches = 500;
  const int batch_size     = 100;
  for (int i = 0; i < warmup_batches; ++i) {
    normalizer.update(make_batch(batch_size));
  }

  // Normalize a fresh large batch (10000 samples) and measure output stats
  const int test_size = 10000;
  auto test_batch = make_batch(test_size);
  auto normalized  = normalizer.normalize(test_batch);

  // Per-feature mean should be ~0
  auto out_mean = normalized.mean(0); // [N_FEATURES]
  for (int f = 0; f < N_FEATURES; ++f) {
    EXPECT_NEAR(out_mean[f].item<float>(), 0.0f, 0.05f)
        << "Feature " << f << ": normalized output mean should be ~0";
  }

  // Per-feature std should be ~1
  auto out_std = normalized.std(0); // [N_FEATURES], unbiased
  for (int f = 0; f < N_FEATURES; ++f) {
    EXPECT_NEAR(out_std[f].item<float>(), 1.0f, 0.05f)
        << "Feature " << f << ": normalized output std should be ~1";
  }
}

// ---- Test 3: incremental vs. single-batch equivalence --------------------
// Verify that many small batch updates give the same running statistics as
// one large batch update. This validates the Chan parallel algorithm.
TEST(RunningNormalizer, IncrementalVsBatch) {
  torch::manual_seed(7);
  torch::NoGradGuard no_grad;

  // Build a fixed dataset once
  const int total_samples = 10000;
  const int small_batch   = 10;
  auto full_data = make_batch(total_samples); // [10000, 4]

  // Normalizer A: one large update
  rl::RunningNormalizer norm_batch;
  norm_batch.update(full_data);

  // Normalizer B: many small updates
  rl::RunningNormalizer norm_incremental;
  for (int i = 0; i < total_samples / small_batch; ++i) {
    norm_incremental.update(full_data.slice(0, i * small_batch, (i + 1) * small_batch));
  }

  // Both should produce identical normalized output for the same input
  auto probe = make_batch(32);
  auto out_batch       = norm_batch.normalize(probe);
  auto out_incremental = norm_incremental.normalize(probe);

  // Element-wise match to float32 precision
  EXPECT_TRUE(torch::allclose(out_batch, out_incremental, /*rtol=*/1e-4, /*atol=*/1e-5))
      << "Batch and incremental normalizers should produce identical output";
}

// ---- Test 4: early return when not enough data ---------------------------
// normalize() should return the input unchanged until at least 2 samples
// have been seen (no valid variance estimate before that).
TEST(RunningNormalizer, EarlyReturnBeforeInitialized) {
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer normalizer;
  EXPECT_FALSE(normalizer.isInitialized());

  auto input = make_batch(4);
  auto output = normalizer.normalize(input);

  // Should be the exact same tensor (no-op)
  EXPECT_TRUE(torch::equal(input, output))
      << "normalize() should return input unchanged before stats are initialized";
}

// ---- Test 5: checkpoint round-trip ---------------------------------------
// Save and load the normalizer state, then verify the loaded normalizer
// produces the same normalized output as the original.
TEST(RunningNormalizer, CheckpointRoundTrip) {
  torch::manual_seed(99);
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer normalizer;
  for (int i = 0; i < 200; ++i) {
    normalizer.update(make_batch(50));
  }

  const std::string path = "/tmp/test_running_normalizer.pt";
  normalizer.save(path);

  rl::RunningNormalizer loaded;
  loaded.load(path);

  ASSERT_TRUE(loaded.isInitialized());

  auto probe = make_batch(16);
  auto out_original = normalizer.normalize(probe);
  auto out_loaded   = loaded.normalize(probe);

  EXPECT_TRUE(torch::allclose(out_original, out_loaded, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "Loaded normalizer should produce identical output to original";
}

// =========================================================================
// scale_only mode tests (return normalization)
// =========================================================================

// ---- Test 6: scale_only preserves the mean --------------------------------
// The defining property of scale_only mode: the mean of the input distribution
// is NOT removed. After normalization the output mean should be ~(true_mean / true_std),
// not ~0.
TEST(RunningNormalizerScaleOnly, MeanPreserved) {
  torch::manual_seed(200);
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer normalizer(1e-8f, /* scale_only = */ true);

  const int batch_size = 100;
  const int n_batches  = 500; // 50000 samples

  for (int i = 0; i < n_batches; ++i) {
    normalizer.update(make_batch(batch_size));
  }

  // Normalize a fresh large batch and check output statistics
  const int test_size = 10000;
  auto test_batch = make_batch(test_size);
  auto normalized  = normalizer.normalize(test_batch);

  auto out_mean = normalized.mean(0); // [N_FEATURES]
  auto out_std  = normalized.std(0);

  for (int f = 0; f < N_FEATURES; ++f) {
    float expected_mean = TRUE_MEAN[f] / TRUE_STD[f];
    // output mean should be ~true_mean / true_std (NOT ~0)
    EXPECT_NEAR(out_mean[f].item<float>(), expected_mean, 0.05f)
        << "Feature " << f << ": scale_only output mean should be ~true_mean/true_std, not 0";

    // output std should still be ~1 (variance is still normalized)
    EXPECT_NEAR(out_std[f].item<float>(), 1.0f, 0.05f)
        << "Feature " << f << ": scale_only output std should be ~1";
  }
}

// ---- Test 7: scale_only vs full — same std, different mean ---------------
// Both modes should produce unit output std. Only the mean differs.
// This test makes the contrast explicit with the same data and seed.
TEST(RunningNormalizerScaleOnly, SameStdDifferentMean) {
  torch::manual_seed(201);
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer full_norm(1e-8f, /* scale_only = */ false);
  rl::RunningNormalizer scale_norm(1e-8f, /* scale_only = */ true);

  const int batch_size = 100;
  const int n_batches  = 500;

  for (int i = 0; i < n_batches; ++i) {
    auto batch = make_batch(batch_size);
    full_norm.update(batch);
    scale_norm.update(batch);
  }

  const int test_size = 10000;
  // use the same seed so both see identical test data
  torch::manual_seed(9999);
  auto test_batch = make_batch(test_size);

  auto out_full  = full_norm.normalize(test_batch);
  auto out_scale = scale_norm.normalize(test_batch);

  auto full_mean  = out_full.mean(0);
  auto scale_mean = out_scale.mean(0);
  auto full_std   = out_full.std(0);
  auto scale_std  = out_scale.std(0);

  for (int f = 0; f < N_FEATURES; ++f) {
    // full mode: mean ~0
    EXPECT_NEAR(full_mean[f].item<float>(), 0.0f, 0.05f)
        << "Feature " << f << ": full mode mean should be ~0";

    // scale_only mode: mean nonzero (only zero if true mean happens to be 0)
    // Here all TRUE_MEAN values are nonzero, so the output mean must differ from 0
    EXPECT_GT(std::abs(scale_mean[f].item<float>()), 0.1f)
        << "Feature " << f << ": scale_only mode mean should be nonzero";

    // both modes: std ~1
    EXPECT_NEAR(full_std[f].item<float>(),  1.0f, 0.05f)
        << "Feature " << f << ": full mode std should be ~1";
    EXPECT_NEAR(scale_std[f].item<float>(), 1.0f, 0.05f)
        << "Feature " << f << ": scale_only mode std should be ~1";
  }
}

// ---- Test 8: scale_only checkpoint round-trip preserves mode -------------
// Saving and loading a scale_only normalizer should restore the flag so that
// the loaded normalizer still does not subtract the mean.
TEST(RunningNormalizerScaleOnly, CheckpointPreservesMode) {
  torch::manual_seed(202);
  torch::NoGradGuard no_grad;

  rl::RunningNormalizer normalizer(1e-8f, /* scale_only = */ true);
  for (int i = 0; i < 200; ++i) {
    normalizer.update(make_batch(50));
  }

  const std::string path = "/tmp/test_running_normalizer_scale_only.pt";
  normalizer.save(path);

  // Load into a default (scale_only=false) instance — the saved flag should override
  rl::RunningNormalizer loaded;
  loaded.load(path);

  auto probe = make_batch(16);
  auto out_original = normalizer.normalize(probe);
  auto out_loaded   = loaded.normalize(probe);

  // Outputs must match exactly (scale_only mode was restored from checkpoint)
  EXPECT_TRUE(torch::allclose(out_original, out_loaded, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "Loaded scale_only normalizer should produce identical output to original";

  // Verify the loaded normalizer does NOT zero-center: its output mean should be nonzero
  auto large_probe = make_batch(5000);
  auto out_large = loaded.normalize(large_probe);
  auto out_mean = out_large.mean(0);
  for (int f = 0; f < N_FEATURES; ++f) {
    EXPECT_GT(std::abs(out_mean[f].item<float>()), 0.1f)
        << "Feature " << f << ": loaded scale_only normalizer must not zero-center output";
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
