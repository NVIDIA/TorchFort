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

void training_test_distributed(const std::string& model_config, std::vector<int> dev_model, std::vector<int> dev_input,
                               std::vector<int64_t> shape, bool should_fail_create, bool should_fail_train,
                               bool should_fail_inference, bool check_result) {

  std::string model_name = generate_random_name(10);
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int rank, size;
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &size));

  // Skip tests if not running with 2 ranks
  if (size != 2) {
    GTEST_SKIP() << "This test requires 2 ranks to run. Skipping.";
  }

#ifdef ENABLE_GPU
  int ngpu;
  cudaGetDeviceCount(&ngpu);
  if (ngpu < 2) {
    GTEST_SKIP() << "This test requires at least 2 GPUs. Skipping.";
  }
#endif

#ifdef ENABLE_GPU
  if (dev_input[rank] != TORCHFORT_DEVICE_CPU) {
    CHECK_CUDA(cudaSetDevice(dev_input[rank]));
  }
#endif

  try {
    CHECK_TORCHFORT(
        torchfort_create_distributed_model(model_name.c_str(), model_config.c_str(), mpi_comm, dev_model[rank]));
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

  if (!check_current_device(dev_input)) FAIL() << "GPU device switched by torchfort_create_distributed_model.";

  auto input = generate_random<float>(shape);
  auto label = generate_random<float>(shape);
  auto output = generate_random<float>(shape);
  float loss_val;

  float* input_ptr = get_data_ptr(input, dev_input[rank]);
  float* label_ptr = get_data_ptr(label, dev_input[rank]);
  float* output_ptr = get_data_ptr(output, dev_input[rank]);

  try {
    CHECK_TORCHFORT(torchfort_train(model_name.c_str(), input_ptr, shape.size(), shape.data(), label_ptr, shape.size(),
                                    shape.data(), &loss_val, TORCHFORT_FLOAT, 0));
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

  if (!check_current_device(dev_input)) FAIL() << "GPU device switched by torchfort_train.";

  try {
    CHECK_TORCHFORT(torchfort_inference(model_name.c_str(), input_ptr, shape.size(), shape.data(), output_ptr,
                                        shape.size(), shape.data(), TORCHFORT_FLOAT, 0));
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
    if (should_fail_train) {
      // pass
    } else {
      FAIL();
    }
  }

  if (!check_current_device(dev_input)) FAIL() << "GPU device switched by torchfort_inference.";

#ifdef ENABLE_GPU
  if (dev_input[rank] != TORCHFORT_DEVICE_CPU) {
    copy_to_host_vector(output, output_ptr);
  }
#endif

  if (check_result) {
    EXPECT_EQ(input, output);
  }

  free_data_ptr(input_ptr, dev_input[rank]);
  free_data_ptr(label_ptr, dev_input[rank]);
  free_data_ptr(output_ptr, dev_input[rank]);
}

TEST(TorchFort, TrainTestDistributedMLPCPUCPU) {
  training_test_distributed("configs/mlp2.yaml", {TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU},
                            {TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU}, {10, 2, 5}, false, false, false, false);
}

#ifdef ENABLE_GPU
TEST(TorchFort, TrainTestDistributedMLPGPUCPU) {
  training_test_distributed("configs/mlp2.yaml", {0, 1}, {TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU}, {10, 2, 5},
                            false, false, false, false);
}
TEST(TorchFort, TrainTestDistributedMLPGPUReverseCPU) {
  training_test_distributed("configs/mlp2.yaml", {1, 0}, {TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU}, {10, 2, 5},
                            false, false, false, false);
}
TEST(TorchFort, TrainTestDistributedMLPCPUGPU) {
  training_test_distributed("configs/mlp2.yaml", {TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU}, {0, 1}, {10, 2, 5},
                            false, false, false, false);
}
TEST(TorchFort, TrainTestDistributedMLPCPUGPUReverse) {
  training_test_distributed("configs/mlp2.yaml", {TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU}, {1, 0}, {10, 2, 5},
                            false, false, false, false);
}
TEST(TorchFort, TrainTestDistributedMLPGPUGPU) {
  training_test_distributed("configs/mlp2.yaml", {0, 1}, {0, 1}, {10, 10}, false, false, false, false);
}
TEST(TorchFort, TrainTestDistributedMLPGPUGPUReverse) {
  training_test_distributed("configs/mlp2.yaml", {0, 1}, {1, 0}, {10, 10}, false, false, false, false);
}
#endif

// Testing expected error cases

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
