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
#include <string>
#include <vector>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torchfort.h"
#include "internal/defines.h"
#include "internal/exceptions.h"
#include "internal/utils.h"

#include "test_utils.h"

void training_test(const std::string& model_config, int dev_model, int dev_input,
                   bool should_fail_create, bool should_fail_train, bool should_fail_inference,
                   bool check_result) {

  std::string model_name = generate_random_name(10);

  try {
    CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), model_config.c_str(), dev_model));
    if (should_fail_create) {
      FAIL() << "This test should fail create call, but did not.";
    }
  } catch (const torchfort::BaseException& e) {
    if (should_fail_create) {
      // pass
    } else {
      FAIL();
    }
  }

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

  std::vector<int64_t> shape = {10, 10};
  auto input = generate_random<float>(shape);
  auto label = generate_random<float>(shape);
  auto output = generate_random<float>(shape);
  float loss_val;

  float *input_ptr = get_data_ptr(input, dev_input);
  float *label_ptr = get_data_ptr(label, dev_input);
  float *output_ptr = get_data_ptr(output, dev_input);

  try {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(),
                                    label_ptr, shape.size(), shape.data(), &loss_val,
                                    TORCHFORT_FLOAT, 0));
    if (should_fail_train) {
      FAIL() << "This test should fail train call, but did not.";
    }
  } catch (const torchfort::BaseException& e) {
    if (should_fail_train) {
      // pass
    } else {
      FAIL();
    }
  }

  try {
    CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(),
                                        output_ptr, shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
    if (should_fail_inference) {
      FAIL() << "This test should fail inference call, but did not.";
    }
  } catch (const torchfort::BaseException& e) {
    if (should_fail_inference) {
      // pass
    } else {
      FAIL();
    }
  }

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    copy_to_host_vector(output, output_ptr);
  }
#endif

  if (check_result) {
    EXPECT_EQ(input, output);
  }

  free_data_ptr(input_ptr, dev_input);
  free_data_ptr(label_ptr, dev_input);
  free_data_ptr(output_ptr, dev_input);

}

void training_test_multiarg(const std::string& model_config, int dev_model, int dev_input,
                            bool use_extra_args) {

  std::string model_name = generate_random_name(10);

 CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), model_config.c_str(), dev_model));

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

  std::vector<int64_t> shape = {10, 10};
  std::vector<std::vector<float>> inputs(2), labels(2), outputs(2);
  for (int i = 0; i < 2; ++i) {
    inputs[i] = generate_random<float>(shape);
    labels[i] = generate_random<float>(shape);
    outputs[i] = generate_random<float>(shape);
  }

  float loss_val;

  std::vector<std::vector<float>> extra_args;
  if (use_extra_args) {
    for (int i = 0; i < 2; ++i) {
      extra_args.push_back(generate_random<float>(shape));
    }
  }

  torchfort_tensor_list_t inputs_tl, labels_tl, outputs_tl;
  CHECK_TORCHFORT(torchfort_tensor_list_create(&inputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&labels_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&outputs_tl));
  std::vector<float*> input_ptrs(2), label_ptrs(2), output_ptrs(2);

  for (int i = 0; i < 2; ++i) {
    input_ptrs[i] = get_data_ptr(inputs[i], dev_input);
    label_ptrs[i] = get_data_ptr(labels[i], dev_input);
    output_ptrs[i] = get_data_ptr(outputs[i], dev_input);
    CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(inputs_tl, input_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(labels_tl, label_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(outputs_tl, output_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
  }

  torchfort_tensor_list_t extra_args_tl;
  std::vector<float*> extra_args_ptrs(2);
  if (use_extra_args) {
    torchfort_tensor_list_create(&extra_args_tl);
    for (int i = 0; i < 2; ++i) {
      extra_args_ptrs[i] = get_data_ptr(extra_args[i], dev_input);
      CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(extra_args_tl, extra_args_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    }
  }

  CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs_tl, labels_tl, &loss_val, (use_extra_args) ? extra_args_tl : nullptr, 0));

  CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), inputs_tl, outputs_tl, 0));

  // Check inference output
  for (int i = 0; i < 2; ++i) {
#ifdef ENABLE_GPU
    if (dev_input != TORCHFORT_DEVICE_CPU) {
      copy_to_host_vector(outputs[i], output_ptrs[i]);
    }
#endif
    EXPECT_EQ(inputs[i], outputs[i]);
  }


  // Check that external data changes reflect in tensor list
  for (int i = 0; i < 2; ++i) {
    auto tmp = generate_random<float>(shape);
    inputs[i].assign(tmp.begin(), tmp.end());
#ifdef ENABLE_GPU
    if (dev_input != TORCHFORT_DEVICE_CPU) {
      copy_from_host_vector(input_ptrs[i], inputs[i]);
    }
#endif
  }

  CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), inputs_tl, outputs_tl, 0));

  for (int i = 0; i < 2; ++i) {
#ifdef ENABLE_GPU
    if (dev_input != TORCHFORT_DEVICE_CPU) {
      copy_to_host_vector(outputs[i], output_ptrs[i]);
    }
#endif
    EXPECT_EQ(inputs[i], outputs[i]);
  }

  for (int i = 0; i < 2; ++i) {
    free_data_ptr(input_ptrs[i], dev_input);
    free_data_ptr(label_ptrs[i], dev_input);
    free_data_ptr(output_ptrs[i], dev_input);
    if (use_extra_args) {
      free_data_ptr(extra_args_ptrs[i], dev_input);
    }
  }

  CHECK_TORCHFORT(torchfort_tensor_list_destroy(inputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(labels_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(outputs_tl));
  if (use_extra_args) {
    CHECK_TORCHFORT(torchfort_tensor_list_destroy(extra_args_tl));
  }

}

void training_test_multiarg_errors(const std::string& model_config) {

  std::string model_name = generate_random_name(10);

  CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), model_config.c_str(), TORCHFORT_DEVICE_CPU));

  torchfort_tensor_list_t inputs, labels, outputs, extra_args;
  CHECK_TORCHFORT(torchfort_tensor_list_create(&inputs));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&labels));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&outputs));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&extra_args));

  float loss_val;
  try {
    CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), nullptr, labels, &loss_val, extra_args, 0));
    CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs, nullptr, &loss_val, extra_args, 0));
    FAIL() << "This test should fail train call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  try {
    CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), nullptr, outputs, 0));
    CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), inputs, nullptr, 0));
    FAIL() << "This test should fail inference call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  CHECK_TORCHFORT(torchfort_tensor_list_destroy(inputs));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(labels));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(outputs));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(extra_args));

}

void training_test_grad_accumulation(const std::string& model_config, int dev_model, int dev_input,
                                     int grad_accumulation_steps) {

  std::string model_name = generate_random_name(10);

  CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), model_config.c_str(), dev_model));

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

  std::vector<int64_t> shape = {10, 10};
  auto input = generate_random<float>(shape);
  auto label = generate_random<float>(shape);
  auto output = generate_random<float>(shape);
  auto output2 = generate_random<float>(shape);
  float loss_val;

  float *input_ptr = get_data_ptr(input, dev_input);
  float *label_ptr = get_data_ptr(label, dev_input);
  float *output_ptr = get_data_ptr(output, dev_input);
  float *output2_ptr = get_data_ptr(output2, dev_input);

  // Get initial output
  CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(),
                                      output_ptr, shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
#ifdef ENABLE_GPU
    if (dev_input != TORCHFORT_DEVICE_CPU) {
      copy_to_host_vector(output, output_ptr);
    }
#endif

  // Run several training steps. Model output should only change after grad_accumulation_steps steps have completed.
  for (int i = 0; i < grad_accumulation_steps; ++i) {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(),
                                    label_ptr, shape.size(), shape.data(), &loss_val,
                                    TORCHFORT_FLOAT, 0));

    CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(),
                                        output2_ptr, shape.size(), shape.data(), TORCHFORT_FLOAT, 0));

#ifdef ENABLE_GPU
    if (dev_input != TORCHFORT_DEVICE_CPU) {
      copy_to_host_vector(output2, output2_ptr);
    }
#endif
    if (i < grad_accumulation_steps - 1) {
      EXPECT_EQ(output, output2);
    } else {
      EXPECT_NE(output, output2);
    }
  }

  free_data_ptr(input_ptr, dev_input);
  free_data_ptr(label_ptr, dev_input);
  free_data_ptr(output_ptr, dev_input);
  free_data_ptr(output2_ptr, dev_input);

}

TEST(TorchFort, TrainTestMLPCPUCPU) {
  training_test("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, false, false, false);
}
TEST(TorchFort, TrainTestTorchScriptCPUCPU) {
  training_test("configs/torchscript.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgCPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraCPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, true);
}

TEST(TorchFort, TrainTestGradAccumulationCPUCPU) {
  training_test_grad_accumulation("configs/mlp2_gradacc.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, 4);
}

#ifdef ENABLE_GPU
TEST(TorchFort, TrainTestMLPGPUCPU) {
  training_test("configs/mlp2.yaml", 0, TORCHFORT_DEVICE_CPU, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPCPUGPU) {
  training_test("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, 0, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPGPUGPU) {
  training_test("configs/mlp2.yaml", 0, 0, false, false, false, false);
}
TEST(TorchFort, TrainTestTorchScriptCPUGPU) {
  training_test("configs/torchscript.yaml", TORCHFORT_DEVICE_CPU, 0, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptGPUCPU) {
  training_test("configs/torchscript.yaml", 0, TORCHFORT_DEVICE_CPU, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptGPUGPU) {
  training_test("configs/torchscript.yaml", 0, 0, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgCPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, 0, false);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 0, TORCHFORT_DEVICE_CPU, false);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 0, 0, false);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraCPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", TORCHFORT_DEVICE_CPU, 0, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", 0, TORCHFORT_DEVICE_CPU, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", 0, 0, true);
}
#endif

// Testing expected error cases
TEST(TorchFort, TrainTestBadConfigName) {
  training_test("configs/blah.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, true, true, true, false);
}
TEST(TorchFort, TrainTestNoOptimizerBlock) {
  training_test("configs/missing_opt.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, true, false, false);
}
TEST(TorchFort, TrainTestNoLossBlock) {
  training_test("configs/missing_loss.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, true, false, false);
}
TEST(TorchFort, TrainTestMultiArgErrors) {
  training_test_multiarg_errors("configs/torchscript_multiarg.yaml");
}


int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
