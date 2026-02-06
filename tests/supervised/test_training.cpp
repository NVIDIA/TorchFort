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
#include <string>
#include <vector>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "internal/defines.h"
#include "internal/exceptions.h"
#include "internal/utils.h"
#include "torchfort.h"

#include "test_utils.h"

void training_test(const std::string& model_config, int dev_model, int dev_input, std::vector<int64_t> shape,
                   bool should_fail_create, bool should_fail_train, bool should_fail_inference, bool check_result,
                   int n_train_steps = 1, int n_inference_steps = 1, int dev_stream = -1) {

  std::string model_name = generate_random_name(10);

#ifdef ENABLE_GPU
  if (dev_model == 1 || dev_input == 1 || dev_stream == 1) {
    int ngpu;
    CHECK_CUDA(cudaGetDeviceCount(&ngpu));
    if (ngpu < 2) {
      GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
    }
  }
#endif

  cudaStream_t stream = nullptr;
#ifdef ENABLE_GPU
  if (dev_stream != -1) {
    CHECK_CUDA(cudaSetDevice(dev_stream));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

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

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_create_model.";

  auto input = generate_random<float>(shape);
  auto label = generate_random<float>(shape);
  auto output = generate_random<float>(shape);
  float loss_val;

  float* input_ptr = get_data_ptr(input, dev_input);
  float* label_ptr = get_data_ptr(label, dev_input);
  float* output_ptr = get_data_ptr(output, dev_input);

#ifdef ENABLE_GPU
  if (stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
#endif

  try {
    for (int i = 0; i < n_train_steps; ++i) {
      auto tmp_input = generate_random<float>(shape);
      auto tmp_label = generate_random<float>(shape);
      std::copy(tmp_input.begin(), tmp_input.end(), input.begin());
      std::copy(tmp_label.begin(), tmp_label.end(), label.begin());
#ifdef ENABLE_GPU
      if (dev_input != TORCHFORT_DEVICE_CPU) {
        copy_from_host_vector(input_ptr, input);
        copy_from_host_vector(label_ptr, label);
      }
#endif
      CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(), label_ptr,
                                      shape.size(), shape.data(), &loss_val, TORCHFORT_FLOAT, stream));
    }
    if (should_fail_train) {
      FAIL() << "This test should fail train call, but did not.";
    }
  } catch (const torchfort::BaseException& e) {
    if (should_fail_train) {
      // pass
    } else {
      FAIL();
    }
  } catch (const c10::Error& e) {
    std::cout << e.what() << std::endl;
    if (should_fail_train) {
      // pass
    } else {
      FAIL();
    }
  }

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_train.";

#ifdef ENABLE_GPU
  if (stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
#endif

  try {
    for (int i = 0; i < n_inference_steps; ++i) {
      auto tmp_input = generate_random<float>(shape);
      std::copy(tmp_input.begin(), tmp_input.end(), input.begin());
#ifdef ENABLE_GPU
      if (dev_input != TORCHFORT_DEVICE_CPU) {
        copy_from_host_vector(input_ptr, input);
      }
#endif
      CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(), output_ptr,
                                          shape.size(), shape.data(), TORCHFORT_FLOAT, stream));
    }
    if (should_fail_inference) {
      FAIL() << "This test should fail inference call, but did not.";
    }
  } catch (const torchfort::BaseException& e) {
    if (should_fail_inference) {
      // pass
    } else {
      FAIL();
    }
  } catch (const c10::Error& e) {
    std::cout << e.what() << std::endl;
    if (should_fail_inference) {
      // pass
    } else {
      FAIL();
    }
  }

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_inference.";

#ifdef ENABLE_GPU
  if (stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
#endif

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

#ifdef ENABLE_GPU
  if (stream) {
    CHECK_CUDA(cudaStreamDestroy(stream));
  }
#endif
}

void training_test_multiarg(const std::string& model_config, int dev_model, int dev_input, bool use_extra_args,
                            bool should_fail_create, bool should_fail_train, bool should_fail_inference,
                            bool check_result, int n_train_steps = 1, int n_inference_steps = 1) {
#ifdef ENABLE_GPU
  if (dev_model == 1 || dev_input == 1) {
    int ngpu;
    cudaGetDeviceCount(&ngpu);
    if (ngpu < 2) {
      GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
    }
  }
#endif

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

  std::string model_name = generate_random_name(10);

  CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), model_config.c_str(), dev_model));

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_create_model.";

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
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(inputs_tl, input_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(labels_tl, label_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(outputs_tl, output_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
  }

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_tensor_list_add_tensor.";

  torchfort_tensor_list_t extra_args_tl;
  std::vector<float*> extra_args_ptrs(2);
  if (use_extra_args) {
    torchfort_tensor_list_create(&extra_args_tl);
    for (int i = 0; i < 2; ++i) {
      extra_args_ptrs[i] = get_data_ptr(extra_args[i], dev_input);
      CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(extra_args_tl, extra_args_ptrs[i], shape.size(), shape.data(),
                                                       TORCHFORT_FLOAT));
    }
  }

  try {
    for (int i = 0; i < n_train_steps; ++i) {
      for (int i = 0; i < 2; ++i) {
        auto tmp_input = generate_random<float>(shape);
        std::copy(tmp_input.begin(), tmp_input.end(), inputs[i].begin());
        auto tmp_label = generate_random<float>(shape);
        std::copy(tmp_label.begin(), tmp_label.end(), labels[i].begin());
        if (use_extra_args) {
          auto tmp_extra_args = generate_random<float>(shape);
          std::copy(tmp_extra_args.begin(), tmp_extra_args.end(), extra_args[i].begin());
        }
#ifdef ENABLE_GPU
        if (dev_input != TORCHFORT_DEVICE_CPU) {
          copy_from_host_vector(input_ptrs[i], inputs[i]);
          copy_from_host_vector(label_ptrs[i], labels[i]);
          if (use_extra_args) {
            copy_from_host_vector(extra_args_ptrs[i], extra_args[i]);
          }
        }
#endif
      }
      CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs_tl, labels_tl, &loss_val,
                                               (use_extra_args) ? extra_args_tl : nullptr, 0));
    }
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

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_train_multiarg.";

  try {
    for (int i = 0; i < n_inference_steps; ++i) {
      for (int i = 0; i < 2; ++i) {
        auto tmp_input = generate_random<float>(shape);
        std::copy(tmp_input.begin(), tmp_input.end(), inputs[i].begin());
#ifdef ENABLE_GPU
        if (dev_input != TORCHFORT_DEVICE_CPU) {
          copy_from_host_vector(input_ptrs[i], inputs[i]);
        }
#endif
      }
      CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), inputs_tl, outputs_tl, 0));
    }
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

  if (!check_current_device(dev_input))
    FAIL() << "GPU device switched by torchfort_inference_multiarg.";

  // Check inference output
  if (check_result) {
    for (int i = 0; i < 2; ++i) {
#ifdef ENABLE_GPU
      if (dev_input != TORCHFORT_DEVICE_CPU) {
        copy_to_host_vector(outputs[i], output_ptrs[i]);
      }
#endif
      EXPECT_EQ(inputs[i], outputs[i]);
    }

    // Check that external data changes reflect in tensor list
    float expected_loss_val = 0.0;
    for (int i = 0; i < 2; ++i) {
      auto tmp = generate_constant<float>(shape, 1);
      inputs[i].assign(tmp.begin(), tmp.end());
      labels[i].assign(tmp.begin(), tmp.end());
      expected_loss_val += 2 * 1.0f * tmp.size();
      if (use_extra_args) {
        extra_args[i].assign(tmp.begin(), tmp.end());
        expected_loss_val += 1.0f * tmp.size();
      }
#ifdef ENABLE_GPU
      if (dev_input != TORCHFORT_DEVICE_CPU) {
        copy_from_host_vector(input_ptrs[i], inputs[i]);
        copy_from_host_vector(label_ptrs[i], labels[i]);
        if (use_extra_args) {
          copy_from_host_vector(extra_args_ptrs[i], extra_args[i]);
        }
      }
#endif
    }

    CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs_tl, labels_tl, &loss_val,
                                             (use_extra_args) ? extra_args_tl : nullptr, 0));

    EXPECT_EQ(loss_val, expected_loss_val);
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

  float* input_ptr = get_data_ptr(input, dev_input);
  float* label_ptr = get_data_ptr(label, dev_input);
  float* output_ptr = get_data_ptr(output, dev_input);
  float* output2_ptr = get_data_ptr(output2, dev_input);

  // Get initial output
  CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(), output_ptr,
                                      shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    copy_to_host_vector(output, output_ptr);
  }
#endif

  // Run several training steps. Model output should only change after grad_accumulation_steps steps have completed.
  for (int i = 0; i < grad_accumulation_steps; ++i) {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(), label_ptr, shape.size(),
                                    shape.data(), &loss_val, TORCHFORT_FLOAT, 0));

    CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(), output2_ptr,
                                        shape.size(), shape.data(), TORCHFORT_FLOAT, 0));

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

#ifdef ENABLE_GPU
void training_test_graphs_errors(int dev_input) {

  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }

  std::string model_name = generate_random_name(10);
  CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), "configs/torchscript_graphs.yaml", 0));

  std::vector<int64_t> shape = {10, 2, 10};
  std::vector<int64_t> shape2 = {10, 3, 10};
  auto input = generate_random<float>(shape);
  auto label = generate_random<float>(shape);
  auto output = generate_random<float>(shape);
  float loss_val;

  float* input_ptr = get_data_ptr(input, dev_input);
  float* label_ptr = get_data_ptr(label, dev_input);
  float* output_ptr = get_data_ptr(output, dev_input);

  // Train and run inference for 4 iterations to trigger graph capture
  for (int i = 0; i < 4; ++i) {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(), label_ptr, shape.size(),
                                    shape.data(), &loss_val, TORCHFORT_FLOAT, 0));
    CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(), output_ptr,
                                        shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
  }

  // Change input buffer
  auto input2 = generate_random<float>(shape);
  float* input2_ptr = get_data_ptr(input2, dev_input);
  try {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input2_ptr, shape.size(), shape.data(), label_ptr, shape.size(),
                                    shape.data(), &loss_val, TORCHFORT_FLOAT, 0));
    FAIL() << "This test should fail train call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  // Change label buffer
  auto label2 = generate_random<float>(shape);
  float* label2_ptr = get_data_ptr(label2, dev_input);
  try {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(), label2_ptr, shape.size(),
                                    shape.data(), &loss_val, TORCHFORT_FLOAT, 0));
    FAIL() << "This test should fail train call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  // Change input buffer for inference
  try {
    CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input2_ptr, shape.size(), shape.data(), output_ptr,
                                        shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
    FAIL() << "This test should fail inference call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  free_data_ptr(input_ptr, dev_input);
  free_data_ptr(label_ptr, dev_input);
  free_data_ptr(output_ptr, dev_input);
  free_data_ptr(input2_ptr, dev_input);
  free_data_ptr(label2_ptr, dev_input);
}

void training_test_multiarg_graphs_errors(int dev_input, bool use_extra_args) {

  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }

  std::string model_name = generate_random_name(10);
  if (use_extra_args) {
    CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), "configs/torchscript_multiarg_extra_graphs.yaml", 0));
  } else {
    CHECK_TORCHFORT(torchfort_create_model(model_name.c_str(), "configs/torchscript_multiarg_graphs.yaml", 0));
  }

  std::vector<int64_t> shape = {10, 10};
  std::vector<std::vector<float>> inputs(2), labels(2), outputs(2);
  std::vector<std::vector<float>> inputs2(2), labels2(2);
  for (int i = 0; i < 2; ++i) {
    inputs[i] = generate_random<float>(shape);
    labels[i] = generate_random<float>(shape);
    outputs[i] = generate_random<float>(shape);
    inputs2[i] = generate_random<float>(shape);
    labels2[i] = generate_random<float>(shape);
  }

  float loss_val;

  std::vector<std::vector<float>> extra_args;
  std::vector<std::vector<float>> extra_args2;
  if (use_extra_args) {
    for (int i = 0; i < 2; ++i) {
      extra_args.push_back(generate_random<float>(shape));
      extra_args2.push_back(generate_random<float>(shape));
    }
  }

  torchfort_tensor_list_t inputs_tl, labels_tl, outputs_tl;
  torchfort_tensor_list_t inputs2_tl, labels2_tl, outputs2_tl;
  CHECK_TORCHFORT(torchfort_tensor_list_create(&inputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&labels_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&outputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&inputs2_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&labels2_tl));

  std::vector<float*> input_ptrs(2), label_ptrs(2), output_ptrs(2);
  std::vector<float*> input2_ptrs(2), label2_ptrs(2);

  for (int i = 0; i < 2; ++i) {
    input_ptrs[i] = get_data_ptr(inputs[i], dev_input);
    label_ptrs[i] = get_data_ptr(labels[i], dev_input);
    output_ptrs[i] = get_data_ptr(outputs[i], dev_input);
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(inputs_tl, input_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(labels_tl, label_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(outputs_tl, output_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));

    input2_ptrs[i] = get_data_ptr(inputs2[i], dev_input);
    label2_ptrs[i] = get_data_ptr(labels2[i], dev_input);
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(inputs2_tl, input2_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(
        torchfort_tensor_list_add_tensor(labels2_tl, label2_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
  }

  torchfort_tensor_list_t extra_args_tl;
  torchfort_tensor_list_t extra_args2_tl;
  std::vector<float*> extra_args_ptrs(2);
  std::vector<float*> extra_args2_ptrs(2);
  if (use_extra_args) {
    torchfort_tensor_list_create(&extra_args_tl);
    torchfort_tensor_list_create(&extra_args2_tl);
    for (int i = 0; i < 2; ++i) {
      extra_args_ptrs[i] = get_data_ptr(extra_args[i], dev_input);
      CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(extra_args_tl, extra_args_ptrs[i], shape.size(), shape.data(),
                                                       TORCHFORT_FLOAT));
      extra_args2_ptrs[i] = get_data_ptr(extra_args2[i], dev_input);
      CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(extra_args2_tl, extra_args2_ptrs[i], shape.size(), shape.data(),
                                                       TORCHFORT_FLOAT));
    }
  }

  // Train and run inference for 4 iterations to trigger graph capture
  for (int i = 0; i < 4; ++i) {
    CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs_tl, labels_tl, &loss_val,
                                             (use_extra_args) ? extra_args_tl : nullptr, 0));
    CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), inputs_tl, outputs_tl, 0));
  }

  // Change input buffer
  try {
    CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs2_tl, labels_tl, &loss_val,
                                             (use_extra_args) ? extra_args_tl : nullptr, 0));
    FAIL() << "This test should fail train call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  // Change label buffer
  try {
    CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs_tl, labels2_tl, &loss_val,
                                             (use_extra_args) ? extra_args_tl : nullptr, 0));
    FAIL() << "This test should fail train call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  // Change extra args buffer
  if (use_extra_args) {
    try {
      CHECK_TORCHFORT(torchfort_train_multiarg(model_name.c_str(), inputs_tl, labels_tl, &loss_val,
                                               (use_extra_args) ? extra_args2_tl : nullptr, 0));
      FAIL() << "This test should fail train call, but did not.";
    } catch (const torchfort::BaseException& e) {
      // pass
    }
  }

  // Change input buffer for inference
  try {
    CHECK_TORCHFORT(torchfort_inference_multiarg(model_name.c_str(), inputs2_tl, outputs_tl, 0));
    FAIL() << "This test should fail inference call, but did not.";
  } catch (const torchfort::BaseException& e) {
    // pass
  }

  for (int i = 0; i < 2; ++i) {
    free_data_ptr(input_ptrs[i], dev_input);
    free_data_ptr(label_ptrs[i], dev_input);
    free_data_ptr(input2_ptrs[i], dev_input);
    free_data_ptr(label2_ptrs[i], dev_input);
    if (use_extra_args) {
      free_data_ptr(extra_args_ptrs[i], dev_input);
      free_data_ptr(extra_args2_ptrs[i], dev_input);
    }
  }
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(inputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(labels_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(outputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(inputs2_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(labels2_tl));
  if (use_extra_args) {
    CHECK_TORCHFORT(torchfort_tensor_list_destroy(extra_args_tl));
    CHECK_TORCHFORT(torchfort_tensor_list_destroy(extra_args2_tl));
  }
}
#endif

TEST(TorchFort, TrainTestMLPCPUCPU) {
  training_test("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 2, 5}, false, false, false,
                false);
}
TEST(TorchFort, TrainTestMLPCPUCPUNoFlatten) {
  training_test("configs/mlp3.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 2, 10}, false, false, false,
                false);
}
TEST(TorchFort, TrainTestMLPCPUCPU1DNoFlatten) {
  training_test("configs/mlp3.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10}, false, false, false, false);
}
TEST(TorchFort, TrainTestTorchScriptCPUCPU) {
  training_test("configs/torchscript.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 2, 10}, false, false,
                false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgCPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, false,
                         false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraCPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, true,
                         false, false, false, true);
}

TEST(TorchFort, TrainTestGradAccumulationCPUCPU) {
  training_test_grad_accumulation("configs/mlp2_gradacc.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, 4);
}

#ifdef ENABLE_GPU
TEST(TorchFort, TrainTestMLPGPUCPU) {
  training_test("configs/mlp2.yaml", 0, TORCHFORT_DEVICE_CPU, {10, 2, 5}, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPCPUGPU) {
  training_test("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, 0, {10, 2, 5}, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPGPUGPU) { training_test("configs/mlp2.yaml", 0, 0, {10, 10}, false, false, false, false); }
TEST(TorchFort, TrainTestMLPGPUGPUStream) {
  training_test("configs/mlp2.yaml", 0, 0, {10, 10}, false, false, false, false, 0);
}

TEST(TorchFort, TrainTestMLPGPU1CPU) {
  training_test("configs/mlp2.yaml", 1, TORCHFORT_DEVICE_CPU, {10, 2, 5}, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPCPUGPU1) {
  training_test("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, 1, {10, 2, 5}, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPGPU0GPU1) {
  training_test("configs/mlp2.yaml", 0, 1, {10, 10}, false, false, false, false);
}
TEST(TorchFort, TrainTestMLPGPU1GPU0) {
  training_test("configs/mlp2.yaml", 1, 0, {10, 10}, false, false, false, false);
}
TEST(TorchFort, TrainTestTorchScriptCPUGPU) {
  training_test("configs/torchscript.yaml", TORCHFORT_DEVICE_CPU, 0, {10, 2, 10}, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptGPUCPU) {
  training_test("configs/torchscript.yaml", 0, TORCHFORT_DEVICE_CPU, {10, 2, 10}, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptGPUGPU) {
  training_test("configs/torchscript.yaml", 0, 0, {10, 2, 10}, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptGPUCPUGraphs) {
  training_test("configs/torchscript_graphs.yaml", 0, TORCHFORT_DEVICE_CPU, {10, 2, 10}, false, true, true, false, 5,
                5);
}
TEST(TorchFort, TrainTestTorchScriptGPUGPUGraphs) {
  training_test("configs/torchscript_graphs.yaml", 0, 0, {10, 2, 10}, false, false, false, true, 5, 5);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgCPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, 0, false, false, false, false,
                         true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 0, TORCHFORT_DEVICE_CPU, false, false, false, false,
                         true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 0, 0, false, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPUCPUGraphs) {
  training_test_multiarg("configs/torchscript_multiarg_graphs.yaml", 0, TORCHFORT_DEVICE_CPU, false, false, true, true,
                         false, 5, 5);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPUGPUGraphs) {
  training_test_multiarg("configs/torchscript_multiarg_graphs.yaml", 0, 0, false, false, false, false, true, 5, 5);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgCPUGPU1) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, 1, false, false, false, false,
                         true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPU1CPU) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 1, TORCHFORT_DEVICE_CPU, false, false, false, false,
                         true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPU0GPU1) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 0, 1, false, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgGPU1GPU0) {
  training_test_multiarg("configs/torchscript_multiarg.yaml", 1, 0, false, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraCPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", TORCHFORT_DEVICE_CPU, 0, true, false, false, false,
                         true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGPUCPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", 0, TORCHFORT_DEVICE_CPU, true, false, false, false,
                         true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGPUGPU) {
  training_test_multiarg("configs/torchscript_multiarg_extra.yaml", 0, 0, true, false, false, false, true);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGPUCPUGraphs) {
  training_test_multiarg("configs/torchscript_multiarg_extra_graphs.yaml", 0, TORCHFORT_DEVICE_CPU, true, false, true,
                         true, false, 5, 5);
}
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGPUGPUGraphs) {
  training_test_multiarg("configs/torchscript_multiarg_extra_graphs.yaml", 0, 0, true, false, false, false, true, 5, 5);
}
#endif

// Testing expected error cases
TEST(TorchFort, TrainTestBadConfigName) {
  training_test("configs/blah.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 10}, true, true, true, false);
}
TEST(TorchFort, TrainTestNoOptimizerBlock) {
  training_test("configs/missing_opt.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 10}, false, true, false,
                false);
}
TEST(TorchFort, TrainTestNoLossBlock) {
  training_test("configs/missing_loss.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 10}, false, true, false,
                false);
}
TEST(TorchFort, TrainTestMultiArgErrors) { training_test_multiarg_errors("configs/torchscript_multiarg.yaml"); }
TEST(TorchFort, TrainTestMultiArgMLPError) {
  training_test_multiarg("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, false, true, true,
                         false);
}
TEST(TorchFort, TrainTestMLPCPUCPUNoFlattenDimError) {
  training_test("configs/mlp3.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10, 2, 5}, false, true, true, false);
}
TEST(TorchFort, TrainTestMLPCPUCPU1DDimError) {
  training_test("configs/mlp2.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, {10}, false, true, true, false);
}

#ifdef ENABLE_GPU
TEST(TorchFort, TrainTestMLPGPUGPUStreamWrongDeviceError) {
  training_test("configs/mlp2.yaml", 0, 0, {10, 10}, false, true, true, false, 1, 1, 1);
}
TEST(TorchFort, TrainTestTorchScriptGraphsErrors) { training_test_graphs_errors(0); }
TEST(TorchFort, TrainTestTorchScriptMultiArgGraphsErrors) { training_test_multiarg_graphs_errors(0, false); }
TEST(TorchFort, TrainTestTorchScriptMultiArgExtraGraphsErrors) { training_test_multiarg_graphs_errors(0, true); }
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
