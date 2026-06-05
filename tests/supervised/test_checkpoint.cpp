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
#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include <filesystem>
#include <string>
#include <vector>

#include "internal/defines.h"
#include "internal/utils.h"
#include "torchfort.h"
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

  for (int i = 0; i < 5; ++i) {

    // generate random input
    {
      torch::NoGradGuard no_grad;
      data.uniform_(0., 1.);
      label.uniform_(0., 1.);
    }

    torchfort_train("mlp", data.data_ptr<float>(), data_shape.size(), data_shape.data(), label.data_ptr<float>(),
                    label_shape.size(), label_shape.data(), &loss, TORCHFORT_FLOAT, 0);
  }

  // compute output
  torchfort_inference("mlp", data.data_ptr<float>(), data_shape.size(), data_shape.data(), output.data_ptr<float>(),
                      label_shape.size(), label_shape.data(), TORCHFORT_FLOAT, 0);

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
  torchfort_inference("mlp_restore", data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                      output.data_ptr<float>(), label_shape.size(), label_shape.data(), TORCHFORT_FLOAT, 0);

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
  torchfort_train("mlp", data.data_ptr<float>(), data_shape.size(), data_shape.data(), label.data_ptr<float>(),
                  label_shape.size(), label_shape.data(), &loss, TORCHFORT_FLOAT, 0);

  torchfort_inference("mlp", data.data_ptr<float>(), data_shape.size(), data_shape.data(), output.data_ptr<float>(),
                      label_shape.size(), label_shape.data(), TORCHFORT_FLOAT, 0);

  output_before = output.clone().to(cpu_dev);

  // new model
  data = data.to(second_dev);
  label = label.to(second_dev);
  output = output.to(second_dev);
  torchfort_train("mlp_restore", data.data_ptr<float>(), data_shape.size(), data_shape.data(), label.data_ptr<float>(),
                  label_shape.size(), label_shape.data(), &loss, TORCHFORT_FLOAT, 0);

  {
    torch::NoGradGuard no_grad;
    output.zero_();
  }
  torchfort_inference("mlp_restore", data.data_ptr<float>(), data_shape.size(), data_shape.data(),
                      output.data_ptr<float>(), label_shape.size(), label_shape.data(), TORCHFORT_FLOAT, 0);

  output_after = output.clone().to(cpu_dev);

  // compute deviation
  mean_diff = torch::mean(torch::abs(output_after - output_before)).item<float>();
  EXPECT_NEAR(mean_diff, 0., 1e-6);
}

TEST(TorchFort, CheckpointSaveRestoreCPUtoCPU) { checkpoint_save_restore(TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU); }

std::string device_suffix(int device) {
  return device == TORCHFORT_DEVICE_CPU ? "cpu" : "gpu" + std::to_string(device);
}

void load_model_mlp_save_restore(int first_device, int second_device) {
  torch::manual_seed(666);

  const std::string model_file =
      "/tmp/torchfort_load_model_mlp_" + device_suffix(first_device) + "_to_" + device_suffix(second_device) + ".pt";
  std::filesystem::remove(model_file);

  torch::Device cpu_dev = torchfort::get_device(TORCHFORT_DEVICE_CPU);
  torch::Device first_dev = torchfort::get_device(first_device);
  torch::Device second_dev = torchfort::get_device(second_device);
  std::vector<int64_t> input_shape{4, 32};
  std::vector<int64_t> label_shape{4, 1};
  auto source_opts = torch::TensorOptions().device(first_dev).dtype(torch::kFloat32);
  auto restore_opts = torch::TensorOptions().device(second_dev).dtype(torch::kFloat32);
  auto input = torch::ones(input_shape, source_opts);
  auto label_source = torch::zeros(label_shape, source_opts);
  auto output = torch::empty(label_shape, source_opts);
  float loss;

  CHECK_TORCHFORT(torchfort_create_model("mlp_src", "configs/mlp.yaml", first_device));

  for (int i = 0; i < 20; ++i) {
    CHECK_TORCHFORT(torchfort_train("mlp_src", input.data_ptr<float>(), input_shape.size(), input_shape.data(),
                                    label_source.data_ptr<float>(), label_shape.size(), label_shape.data(), &loss,
                                    TORCHFORT_FLOAT, 0));
  }

  CHECK_TORCHFORT(torchfort_inference("mlp_src", input.data_ptr<float>(), input_shape.size(), input_shape.data(),
                                      output.data_ptr<float>(), label_shape.size(), label_shape.data(), TORCHFORT_FLOAT,
                                      0));
  auto output_source = output.clone().to(cpu_dev);

  CHECK_TORCHFORT(torchfort_save_model("mlp_src", model_file.c_str()));

  CHECK_TORCHFORT(torchfort_create_model("mlp_restore_load_model", "configs/mlp.yaml", second_device));
  CHECK_TORCHFORT(torchfort_load_model("mlp_restore_load_model", model_file.c_str()));

  input = input.to(second_dev);
  auto label_restore = torch::full(label_shape, 2.0f, restore_opts);
  output = torch::empty(label_shape, restore_opts);

  CHECK_TORCHFORT(torchfort_inference("mlp_restore_load_model", input.data_ptr<float>(), input_shape.size(),
                                      input_shape.data(), output.data_ptr<float>(), label_shape.size(),
                                      label_shape.data(), TORCHFORT_FLOAT, 0));
  auto output_loaded = output.clone().to(cpu_dev);

  float mean_diff = torch::mean(torch::abs(output_loaded - output_source)).item<float>();
  EXPECT_NEAR(mean_diff, 0.0f, 1e-6f);

  CHECK_TORCHFORT(torchfort_train("mlp_restore_load_model", input.data_ptr<float>(), input_shape.size(),
                                  input_shape.data(), label_restore.data_ptr<float>(), label_shape.size(),
                                  label_shape.data(), &loss, TORCHFORT_FLOAT, 0));

  CHECK_TORCHFORT(torchfort_inference("mlp_restore_load_model", input.data_ptr<float>(), input_shape.size(),
                                      input_shape.data(), output.data_ptr<float>(), label_shape.size(),
                                      label_shape.data(), TORCHFORT_FLOAT, 0));

  mean_diff = torch::mean(torch::abs(output.clone().to(cpu_dev) - output_loaded)).item<float>();
  EXPECT_GT(mean_diff, 1e-6f);

  std::filesystem::remove(model_file);
}

TEST(TorchFort, LoadModelMLPCPUtoCPU) { load_model_mlp_save_restore(TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU); }

void load_model_torchscript_save_restore(int first_device, int second_device) {
  torch::manual_seed(666);

  const std::string model_file = "/tmp/torchfort_load_model_torchscript_" + device_suffix(first_device) + "_to_" +
                                 device_suffix(second_device) + ".pt";
  std::filesystem::remove(model_file);

  torch::Device cpu_dev = torchfort::get_device(TORCHFORT_DEVICE_CPU);
  torch::Device first_dev = torchfort::get_device(first_device);
  torch::Device second_dev = torchfort::get_device(second_device);
  std::vector<int64_t> shape{4, 2, 10};
  auto source_opts = torch::TensorOptions().device(first_dev).dtype(torch::kFloat32);
  auto restore_opts = torch::TensorOptions().device(second_dev).dtype(torch::kFloat32);
  auto input = torch::ones(shape, source_opts);
  auto label_source = torch::zeros(shape, source_opts);
  auto output = torch::empty(shape, source_opts);
  float loss;

  CHECK_TORCHFORT(torchfort_create_model("torchscript_src", "configs/torchscript_trainable.yaml", first_device));

  for (int i = 0; i < 20; ++i) {
    CHECK_TORCHFORT(torchfort_train("torchscript_src", input.data_ptr<float>(), shape.size(), shape.data(),
                                    label_source.data_ptr<float>(), shape.size(), shape.data(), &loss, TORCHFORT_FLOAT,
                                    0));
  }

  CHECK_TORCHFORT(torchfort_inference("torchscript_src", input.data_ptr<float>(), shape.size(), shape.data(),
                                      output.data_ptr<float>(), shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
  auto output_source = output.clone().to(cpu_dev);

  CHECK_TORCHFORT(torchfort_save_model("torchscript_src", model_file.c_str()));

  CHECK_TORCHFORT(torchfort_create_model("torchscript_restore", "configs/torchscript_trainable.yaml", second_device));
  CHECK_TORCHFORT(torchfort_load_model("torchscript_restore", model_file.c_str()));

  input = input.to(second_dev);
  auto label_restore = torch::full(shape, 2.0f, restore_opts);
  output = torch::empty(shape, restore_opts);

  CHECK_TORCHFORT(torchfort_inference("torchscript_restore", input.data_ptr<float>(), shape.size(), shape.data(),
                                      output.data_ptr<float>(), shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
  auto output_loaded = output.clone().to(cpu_dev);

  float mean_diff = torch::mean(torch::abs(output_loaded - output_source)).item<float>();
  EXPECT_NEAR(mean_diff, 0.0f, 1e-6f);

  CHECK_TORCHFORT(torchfort_train("torchscript_restore", input.data_ptr<float>(), shape.size(), shape.data(),
                                  label_restore.data_ptr<float>(), shape.size(), shape.data(), &loss, TORCHFORT_FLOAT,
                                  0));

  CHECK_TORCHFORT(torchfort_inference("torchscript_restore", input.data_ptr<float>(), shape.size(), shape.data(),
                                      output.data_ptr<float>(), shape.size(), shape.data(), TORCHFORT_FLOAT, 0));

  mean_diff = torch::mean(torch::abs(output.clone().to(cpu_dev) - output_loaded)).item<float>();
  EXPECT_GT(mean_diff, 1e-6f);

  std::filesystem::remove(model_file);
}

TEST(TorchFort, LoadModelTorchScriptCPUtoCPU) {
  load_model_torchscript_save_restore(TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
}

#ifdef ENABLE_GPU
TEST(TorchFort, CheckpointSaveRestoreGPUtoGPU) { checkpoint_save_restore(0, 0); }

TEST(TorchFort, CheckpointSaveRestoreCPUtoGPU) { checkpoint_save_restore(TORCHFORT_DEVICE_CPU, 0); }

TEST(TorchFort, CheckpointSaveRestoreGPUtoCPU) { checkpoint_save_restore(0, TORCHFORT_DEVICE_CPU); }

TEST(TorchFort, CheckpointSaveRestoreGPU0toGPU1) {
  int ngpu;
  cudaGetDeviceCount(&ngpu);
  if (ngpu < 2) {
    GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
  }
  checkpoint_save_restore(0, 1);
}

TEST(TorchFort, LoadModelMLPGPUtoGPU) { load_model_mlp_save_restore(0, 0); }

TEST(TorchFort, LoadModelMLPCPUtoGPU) { load_model_mlp_save_restore(TORCHFORT_DEVICE_CPU, 0); }

TEST(TorchFort, LoadModelMLPGPUtoCPU) { load_model_mlp_save_restore(0, TORCHFORT_DEVICE_CPU); }

TEST(TorchFort, LoadModelMLPGPU0toGPU1) {
  int ngpu;
  cudaGetDeviceCount(&ngpu);
  if (ngpu < 2) {
    GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
  }
  load_model_mlp_save_restore(0, 1);
}

TEST(TorchFort, LoadModelTorchScriptGPUtoGPU) { load_model_torchscript_save_restore(0, 0); }

TEST(TorchFort, LoadModelTorchScriptCPUtoGPU) { load_model_torchscript_save_restore(TORCHFORT_DEVICE_CPU, 0); }

TEST(TorchFort, LoadModelTorchScriptGPUtoCPU) { load_model_torchscript_save_restore(0, TORCHFORT_DEVICE_CPU); }

TEST(TorchFort, LoadModelTorchScriptGPU0toGPU1) {
  int ngpu;
  cudaGetDeviceCount(&ngpu);
  if (ngpu < 2) {
    GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
  }
  load_model_torchscript_save_restore(0, 1);
}
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
