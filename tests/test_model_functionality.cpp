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


TEST(TorchFort, CheckpointSaveRestore) {

  // rng
  torch::manual_seed(666);

  // create model on CPU
  torchfort_create_model("mlp", "configs/mlp.yaml", TORCHFORT_DEVICE_CPU);

  // run a few training steps on random data:
  std::vector<int64_t> data_shape = {8, 32};
  std::vector<int64_t> label_shape = {8, 1};
  torch::Tensor data = torch::empty(data_shape, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  torch::Tensor label = torch::empty(label_shape, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  torch::Tensor output = torch::empty(label_shape, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
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
  
  torch::Tensor output_before = output.clone();

  // save model checkpoint
  torchfort_save_checkpoint("mlp", "/tmp/checkpoint.pt"); 

  // initialize a new model
  torchfort_create_model("mlp_restore", "configs/mlp.yaml", TORCHFORT_DEVICE_CPU);

  // restore old model from checkpoint
  int64_t step_train, step_inference;
  torchfort_load_checkpoint("mlp_restore", "/tmp/checkpoint.pt", &step_train, &step_inference);

  // ensure that the steps are correct
  EXPECT_EQ(step_train, 5);
  EXPECT_EQ(step_inference, 1);

  // do a forward pass and compare against the other one:
  {
    torch::NoGradGuard no_grad;
    output.zero_();
  }
  torchfort_inference("mlp_restore",
                      data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                      output.data_ptr<float>(), label_shape.size(), label_shape.data(),
                      TORCHFORT_FLOAT, 0);
  
  torch::Tensor output_after = output.clone();

  // compute deviation
  float mean_diff = torch::mean(torch::abs(output_after - output_before)).item<float>();
  EXPECT_FLOAT_EQ(mean_diff, 0.);

  // lastly, perform one more optimizer step
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
  
  output_before = output.clone();

  // new model
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
  
  output_after = output.clone();

  // compute deviation
  mean_diff = torch::mean(torch::abs(output_after - output_before)).item<float>();
  EXPECT_FLOAT_EQ(mean_diff, 0.);
  
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
