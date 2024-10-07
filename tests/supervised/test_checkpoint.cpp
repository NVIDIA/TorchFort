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
#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include "torchfort.h"
#include "internal/utils.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

void checkpoint_save_restore(int first_device, int second_device) {

  // rng
  torch::manual_seed(666);

  // create devices:
  torch::Device cpu_dev = torchfort::get_device(TORCHFORT_DEVICE_CPU);
  torch::Device first_dev = torchfort::get_device(first_device);
  torch::Device second_dev = torchfort::get_device(second_device);

  // create model on CPU
  torchfort_create_model("mlp", "configs/mlp.yaml", first_device);

  // run a few training steps on random data:
  std::vector<int64_t> data_shape = {8, 32};
  std::vector<int64_t> label_shape = {8, 1};
  torch::Tensor data = torch::empty(data_shape, torch::TensorOptions().device(first_dev).dtype(torch::kFloat32));
  torch::Tensor label = torch::empty(label_shape, torch::TensorOptions().device(first_dev).dtype(torch::kFloat32));
  torch::Tensor output = torch::empty(label_shape, torch::TensorOptions().device(first_dev).dtype(torch::kFloat32));
  float loss;

  for(int i=0; i<5; ++i) {

    // generate random input
    {
      torch::NoGradGuard no_grad;
      data.uniform_(0., 1.);
      label.uniform_(0., 1.);
    }

    torchfort_train("mlp",
		    data.data_ptr<float>(), data_shape.size(), data_shape.data(),
		    label.data_ptr<float>(), label_shape.size(), label_shape.data(),
		    &loss, TORCHFORT_FLOAT, 0);
  }

  // compute output
  torchfort_inference("mlp",
		      data.data_ptr<float>(), data_shape.size(), data_shape.data(),
		      output.data_ptr<float>(), label_shape.size(), label_shape.data(),
		      TORCHFORT_FLOAT, 0);
  
  torch::Tensor output_before = output.clone().to(cpu_dev);
  auto lrs_before = torchfort::get_current_lrs("mlp");

  // save model checkpoint
  torchfort_save_checkpoint("mlp", "/tmp/checkpoint.pt"); 

  // initialize a new model
  torchfort_create_model("mlp_restore", "configs/mlp.yaml", second_device);

  // restore old model from checkpoint
  int64_t step_train, step_inference;
  torchfort_load_checkpoint("mlp_restore", "/tmp/checkpoint.pt", &step_train, &step_inference);

  // ensure that the steps are correct
  EXPECT_EQ(step_train, 5);
  EXPECT_EQ(step_inference, 1);

  // ensure that the LR is correct
  EXPECT_EQ(lrs_before, torchfort::get_current_lrs("mlp_restore"));

  // do a forward pass and compare against the other one:
  {
    torch::NoGradGuard no_grad;
    output.zero_();
  }
  data = data.to(second_dev);
  output = output.to(second_dev);
  torchfort_inference("mlp_restore",
                      data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                      output.data_ptr<float>(), label_shape.size(), label_shape.data(),
                      TORCHFORT_FLOAT, 0);
  
  torch::Tensor output_after = output.clone().to(cpu_dev);

  // compute deviation
  float mean_diff = torch::mean(torch::abs(output_after - output_before)).item<float>();
  EXPECT_NEAR(mean_diff, 0., 1e-6);

  // lastly, perform one more optimizer step
  data = data.to(first_dev);
  label = label.to(first_dev);
  output = output.to(first_dev);
  {
    torch::NoGradGuard no_grad;
    data.uniform_(0., 1.);
    label.uniform_(0., 1.);
  }

  // old model
  torchfort_train("mlp",
		  data.data_ptr<float>(), data_shape.size(), data_shape.data(),
		  label.data_ptr<float>(), label_shape.size(), label_shape.data(),
		  &loss, TORCHFORT_FLOAT, 0);
  
  torchfort_inference("mlp",
                      data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                      output.data_ptr<float>(), label_shape.size(), label_shape.data(),
                      TORCHFORT_FLOAT, 0);
  
  output_before = output.clone().to(cpu_dev);

  // new model
  data = data.to(second_dev);
  label	= label.to(second_dev);
  output = output.to(second_dev);
  torchfort_train("mlp_restore",
                    data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                    label.data_ptr<float>(), label_shape.size(), label_shape.data(),
                    &loss, TORCHFORT_FLOAT, 0);

  {
    torch::NoGradGuard no_grad;
    output.zero_();
  }
  torchfort_inference("mlp_restore",
                      data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                      output.data_ptr<float>(), label_shape.size(), label_shape.data(),
                      TORCHFORT_FLOAT, 0);
  
  output_after = output.clone().to(cpu_dev);

  // compute deviation
  mean_diff = torch::mean(torch::abs(output_after - output_before)).item<float>();
  EXPECT_NEAR(mean_diff, 0., 1e-6);
  
}


TEST(TorchFort, CheckpointSaveRestoreCPUtoCPU) {
  checkpoint_save_restore(TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
}

#ifdef ENABLE_GPU
TEST(TorchFort, CheckpointSaveRestoreGPUtoGPU) {
  checkpoint_save_restore(0, 0);
}

TEST(TorchFort, CheckpointSaveRestoreCPUtoGPU) {
  checkpoint_save_restore(TORCHFORT_DEVICE_CPU, 0);
}

TEST(TorchFort, CheckpointSaveRestoreGPUtoCPU) {
  checkpoint_save_restore(0, TORCHFORT_DEVICE_CPU);
}

TEST(TorchFort, CheckpointSaveRestoreGPU0toGPU1) {
  int ngpu;
  cudaGetDeviceCount(&ngpu);
  if (ngpu < 2) {
    GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
  }
  checkpoint_save_restore(0, 1);
}
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
