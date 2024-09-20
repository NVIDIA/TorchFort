/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "torchfort.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace torchfort;


TEST(TorchFort, CheckpointSaveRestore) {

  // rng
  torch::manual_seed(666);

  // create model on CPU
  torchfort_create_model("mlp_cpu", "configs/mlp.yaml", TORCHFORT_DEVICE_CPU);

  // run a few training steps on random data:
  torch::tensor data = torch.empty((8, 32), torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  torch::tensor label = torch.empty((8, 1), torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  float* loss;

  for(int i=0; i<5; ++i) {

    // generate random input
    {
      torch::NoGradGuard no_grad;
      data.uniform_(0., 1.);
      label.uniform(0., 1.);
    }

    torchfort_train("mlp_cpu", data.data(), 2, {8, 32}, label.data(), 2, {8, 1}, &loss, TORCHFORT_FLOAT);
    std::cout << "loss" << loss;
  }

  // save model
  /torchfort_save_checkpoint("mlp_cpu", "/tmp/checkpoint_cpu.pt");
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
