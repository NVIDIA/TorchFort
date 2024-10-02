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
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
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

template <typename T>
std::vector<T> generate_random(const std::vector<int64_t>& shape) {

  int64_t num_values = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  std::vector<T> data(num_values);

  std::mt19937 generator;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);
  std::uniform_real_distribution<T> dist((T)0, (T)1);

  auto r = [&]() {
    return dist(generator);
  };

  std::generate(data.begin(), data.end(), r);

  return data;

}

template <typename T>
T* get_data_ptr(std::vector<T>& data, int dev) {
  T* data_ptr;
#ifdef ENABLE_GPU
  if (dev == TORCHFORT_DEVICE_CPU) {
    data_ptr = data.data();
  } else {
    CHECK_CUDA(cudaMalloc(&data_ptr, data.size() * sizeof(T(0))));
    CHECK_CUDA(cudaMemcpy(data_ptr, data.data(), data.size() * sizeof(T(0)), cudaMemcpyHostToDevice));
  }
#else
  data_ptr = data.data();
#endif

  return data_ptr;
}

template <typename T>
void free_data_ptr(T* data_ptr, int dev) {
#ifdef ENABLE_GPU
  if (dev != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaFree(data_ptr));
  }
#endif
}

template <typename T>
T mse_loss(const std::vector<std::vector<T>>& inputs, const std::vector<std::vector<T>>& labels,
           const std::vector<std::vector<T>>& aux_data) {

  T loss = 0.0;
  for (int i = 0; i < inputs[0].size(); ++i) {
    loss += (inputs[0][i] - labels[0][i]) * (inputs[0][i] - labels[0][i]);
  }

  return loss / inputs[0].size();
}

template <typename T>
T l1_loss(const std::vector<std::vector<T>>& inputs, const std::vector<std::vector<T>>& labels,
          const std::vector<std::vector<T>>& aux_data) {

  T loss = 0.0;
  for (int i = 0; i < inputs[0].size(); ++i) {
    loss += std::abs(inputs[0][i] - labels[0][i]);
  }

  return loss / inputs[0].size();
}

template <typename T>
T torchscript_loss(const std::vector<std::vector<T>>& inputs, const std::vector<std::vector<T>>& labels,
                   const std::vector<std::vector<T>>& aux_data) {

  T loss = 0.0;
  if (aux_data.size() == 0) {
    for (int j = 0; j < inputs.size(); ++j) {
      for (int i = 0; i < inputs[j].size(); ++i) {
        loss += inputs[j][i] + labels[j][i];
      }
    }
    loss /= (2.0 * inputs.size() * inputs[0].size());
  } else {
    for (int j = 0; j < inputs.size(); ++j) {
      for (int i = 0; i < inputs[j].size(); ++i) {
        loss += inputs[j][i] + labels[j][i] + aux_data[j][i];
      }
    }
    loss /= (3.0 * inputs.size() * inputs[0].size());
  }

  return loss;
}

template<typename F>
void loss_test(const std::string& model_config, int dev_model, int dev_input, bool should_fail_train, F func) {

  CHECK_TORCHFORT(torchfort_create_model("mymodel", model_config.c_str(), dev_model));

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

  std::vector<int64_t> shape = {10, 10};
  auto input = generate_random<float>(shape);
  auto label = generate_random<float>(shape);
  float loss_val;

  float *input_ptr = get_data_ptr(input, dev_input);
  float *label_ptr = get_data_ptr(label, dev_input);

  try {
    CHECK_TORCHFORT(torchfort_train("mymodel", input_ptr, shape.size(), shape.data(),
                                    label_ptr, shape.size(), shape.data(), &loss_val,
                                    TORCHFORT_FLOAT, 0));
  } catch (const torchfort::BaseException& e) {
    if (should_fail_train) {
      return;
    } else {
      FAIL();
    }
  }

  EXPECT_NEAR(loss_val, func({input}, {label}, {}), 1e-6);

  free_data_ptr(input_ptr, dev_input);
  free_data_ptr(label_ptr, dev_input);

}

template<typename F>
void loss_test_multiarg(const std::string& model_config, int dev_model, int dev_input, bool should_fail_train,
                        bool use_aux_data, F func) {

  CHECK_TORCHFORT(torchfort_create_model("mymodel", model_config.c_str(), dev_model));

#ifdef ENABLE_GPU
  if (dev_input != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input));
  }
#endif

  std::vector<int64_t> shape = {10, 10};
  std::vector<std::vector<float>> inputs(2), labels(2);
  for (int i = 0; i < 2; ++i) {
    inputs[i] = generate_random<float>(shape);
    labels[i] = generate_random<float>(shape);
  }

  float loss_val;

  std::vector<std::vector<float>> aux_data;
  if (use_aux_data) {
    for (int i = 0; i < 2; ++i) {
      aux_data.push_back(generate_random<float>(shape));
    }
  }

  torchfort_tensor_list_t inputs_tl;
  torchfort_tensor_list_t labels_tl;
  CHECK_TORCHFORT(torchfort_tensor_list_create(&inputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_create(&labels_tl));
  std::vector<float*> input_ptrs(2), label_ptrs(2);

  for (int i = 0; i < 2; ++i) {
    input_ptrs[i] = get_data_ptr(inputs[i], dev_input);
    label_ptrs[i] = get_data_ptr(labels[i], dev_input);
    CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(inputs_tl, input_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(labels_tl, label_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
  }

  torchfort_tensor_list_t aux_data_tl;
  std::vector<float*> aux_data_ptrs(2);
  if (use_aux_data) {
    torchfort_tensor_list_create(&aux_data_tl);
    for (int i = 0; i < 2; ++i) {
      aux_data_ptrs[i] = get_data_ptr(aux_data[i], dev_input);
      CHECK_TORCHFORT(torchfort_tensor_list_add_tensor(aux_data_tl, aux_data_ptrs[i], shape.size(), shape.data(), TORCHFORT_FLOAT));
    }
  }

  try {
    CHECK_TORCHFORT(torchfort_train_multiarg("mymodel", inputs_tl, labels_tl, &loss_val, (use_aux_data) ? aux_data_tl : nullptr, 0));
  } catch (const torchfort::BaseException& e) {
    if (should_fail_train) {
      return;
    } else {
      FAIL();
    }
  }

  EXPECT_NEAR(loss_val, func(inputs, labels, aux_data), 1e-6);

  for (int i = 0; i < 2; ++i) {
    free_data_ptr(input_ptrs[i], dev_input);
    free_data_ptr(label_ptrs[i], dev_input);
    if (use_aux_data) {
      free_data_ptr(aux_data_ptrs[i], dev_input);
    }
  }

  CHECK_TORCHFORT(torchfort_tensor_list_destroy(inputs_tl));
  CHECK_TORCHFORT(torchfort_tensor_list_destroy(labels_tl));
  if (use_aux_data) {
    CHECK_TORCHFORT(torchfort_tensor_list_destroy(aux_data_tl));
  }

}

TEST(TorchFort, MSELossCPUCPU) {
  loss_test("configs/mse.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, mse_loss<float>);
}
TEST(TorchFort, L1LossCPUCPU) {
  loss_test("configs/l1.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, l1_loss<float>);
}
TEST(TorchFort, TorchScriptLossCPUCPU) {
  loss_test("configs/torchscript.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiArgCPUCPU) {
  loss_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, false, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiArgAuxDataCPUCPU) {
  loss_test_multiarg("configs/torchscript_multiarg_aux.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, false, true, torchscript_loss<float>);
}

#ifdef ENABLE_GPU
TEST(TorchFort, MSELossCPUGPU) {
  loss_test("configs/mse.yaml", TORCHFORT_DEVICE_CPU, 0, false, mse_loss<float>);
}
TEST(TorchFort, MSELossGPUCPU) {
  loss_test("configs/mse.yaml", 0, TORCHFORT_DEVICE_CPU, false, mse_loss<float>);
}
TEST(TorchFort, MSELossGPUGPU) {
  loss_test("configs/mse.yaml", 0, 0, false, mse_loss<float>);
}
TEST(TorchFort, L1LossCPUGPU) {
  loss_test("configs/l1.yaml", TORCHFORT_DEVICE_CPU, 0, false, l1_loss<float>);
}
TEST(TorchFort, L1LossGPUCPU) {
  loss_test("configs/l1.yaml", 0, TORCHFORT_DEVICE_CPU, false, l1_loss<float>);
}
TEST(TorchFort, L1LossGPUGPU) {
  loss_test("configs/l1.yaml", 0, 0, false, l1_loss<float>);
}
TEST(TorchFort, TorchScriptLossCPUGPU) {
  loss_test("configs/torchscript.yaml", TORCHFORT_DEVICE_CPU, 0, false, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossGPUCPU) {
  loss_test("configs/torchscript.yaml", 0, TORCHFORT_DEVICE_CPU, false, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossGPUGPU) {
  loss_test("configs/torchscript.yaml", 0, 0, false, torchscript_loss<float>);
}

TEST(TorchFort, TorchScriptLossMultiArgCPUGPU) {
  loss_test_multiarg("configs/torchscript_multiarg.yaml", 0, TORCHFORT_DEVICE_CPU, false, false, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiArgGPUCPU) {
  loss_test_multiarg("configs/torchscript_multiarg.yaml", TORCHFORT_DEVICE_CPU, 0, false, false, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiArgGPUGPU) {
  loss_test_multiarg("configs/torchscript_multiarg.yaml", 0, 0, false, false, torchscript_loss<float>);
}

TEST(TorchFort, TorchScriptLossMultiArgAuxDataCPUGPU) {
  loss_test_multiarg("configs/torchscript_multiarg_aux.yaml", TORCHFORT_DEVICE_CPU, 0, false, true, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiArgAuxDataGPUCPU) {
  loss_test_multiarg("configs/torchscript_multiarg_aux.yaml", 0, TORCHFORT_DEVICE_CPU, false, true, torchscript_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiArgAuxDataGPUGPU) {
  loss_test_multiarg("configs/torchscript_multiarg_aux.yaml", 0, 0, false, true, torchscript_loss<float>);
}
#endif

// Testing expected error cases
TEST(TorchFort, MSELossMultiArgError) {
  loss_test_multiarg("configs/mse_multiarg.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, true, false, mse_loss<float>);
}
TEST(TorchFort, L1LossMultiArgError) {
  loss_test_multiarg("configs/l1_multiarg.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, true, false, mse_loss<float>);
}
TEST(TorchFort, TorchScriptLossMultiOutputError) {
  loss_test("configs/torchscript_multiout.yaml", TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU, true, torchscript_loss<float>);
}


int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
