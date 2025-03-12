/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <any>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif
#include <torch/script.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "internal/base_model.h"
#include "internal/exceptions.h"
#include "internal/logging.h"
#include "internal/model_pack.h"
#include "internal/model_wrapper.h"
#include "internal/models.h"
#include "internal/setup.h"
#include "internal/tensor_list.h"
#include "internal/training.h"
#include "internal/utils.h"
#include "torchfort.h"

namespace torchfort {
// Global variables
std::unordered_map<std::string, ModelPack> models;
} // namespace torchfort

torchfort_result_t torchfort_set_cudnn_benchmark(const bool flag) {
  at::globalContext().setBenchmarkCuDNN(flag);
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_set_cuda_allow_tf32(const bool flag) {
  at::globalContext().setAllowTF32CuBLAS(flag);
  at::globalContext().setAllowTF32CuDNN(flag);
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_set_manual_seed(const int seed) {
  torch::manual_seed(static_cast<uint64_t>(seed));
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_set_cuda_manual_seed(const int seed) {
  torch::cuda::manual_seed(static_cast<uint64_t>(seed));
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_create_model(const char* name, const char* config_fname, int device) {
  using namespace torchfort;

  try {
    YAML::Node config;
    try {
      config = YAML::LoadFile(config_fname);
    } catch (const std::exception& e) {
      THROW_INVALID_USAGE("Model configuration file failed to load.");
    }

    // Setting up model
    if (config["model"]) {
      models[name].model = get_model(config["model"]);
      models[name].model->to(get_device(device));
    } else {
      THROW_INVALID_USAGE("Missing model block in configuration file.");
    }

    // Setting up loss
    if (config["loss"]) {
      models[name].loss = get_loss(config["loss"]);
    }

    // Setting up optimizer
    if (config["optimizer"]) {
      models[name].optimizer = get_optimizer(config["optimizer"], models[name].model);

      if (config["optimizer"]["general"]) {
        auto params = get_params(config["optimizer"]["general"]);
        std::set<std::string> supported_params{"grad_accumulation_steps"};
        check_params(supported_params, params.keys());
        try {
          models[name].grad_accumulation_steps = params.get_param<int>("grad_accumulation_steps")[0];
        } catch (std::out_of_range) {
          // default
        }
      }
    }

    // Setting up lr_scheduler
    if (config["lr_scheduler"]) {
      if (models[name].optimizer) {
        models[name].lr_scheduler = get_lr_scheduler(config["lr_scheduler"], models[name].optimizer);
      } else {
        THROW_INVALID_USAGE("LR scheduler defined but no optimizer block found in configuration file.");
      }
    }

    // Setting up general options
    models[name].state = get_state(name, config);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_create_distributed_model(const char* name, const char* config_fname, MPI_Comm mpi_comm,
                                                      int device) {
  using namespace torchfort;

  try {
    torchfort_create_model(name, config_fname, device);

    // Set up distributed communicator
    models[name].comm = std::shared_ptr<Comm>(new Comm(mpi_comm));
    models[name].comm->initialize(models[name].model->device().is_cuda());

    // Broadcast initial model parameters from rank 0
    for (auto& p : models[name].model->parameters()) {
      models[name].comm->broadcast(p, 0);
    }

  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// W&B log function implementations by type
WANDB_LOG_FUNC(int)
WANDB_LOG_FUNC(float)
WANDB_LOG_FUNC(double)

torchfort_result_t torchfort_train(const char* name, void* input, size_t input_dim, int64_t* input_shape, void* label,
                                   size_t label_dim, int64_t* label_shape, void* loss_val, torchfort_datatype_t dtype,
                                   cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      torchfort::train<torchfort::RowMajor>(name, reinterpret_cast<float*>(input), input_dim, input_shape,
                                            reinterpret_cast<float*>(label), label_dim, label_shape,
                                            reinterpret_cast<float*>(loss_val), stream);
      break;
    case TORCHFORT_DOUBLE:
      torchfort::train<torchfort::RowMajor>(name, reinterpret_cast<double*>(input), input_dim, input_shape,
                                            reinterpret_cast<double*>(label), label_dim, label_shape,
                                            reinterpret_cast<double*>(loss_val), stream);
      break;
    default:
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_train_F(const char* name, void* input, size_t input_dim, int64_t* input_shape, void* label,
                                     size_t label_dim, int64_t* label_shape, void* loss_val, torchfort_datatype_t dtype,
                                     cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      torchfort::train<torchfort::ColMajor>(name, reinterpret_cast<float*>(input), input_dim, input_shape,
                                            reinterpret_cast<float*>(label), label_dim, label_shape,
                                            reinterpret_cast<float*>(loss_val), stream);
      break;
    case TORCHFORT_DOUBLE:
      torchfort::train<torchfort::ColMajor>(name, reinterpret_cast<double*>(input), input_dim, input_shape,
                                            reinterpret_cast<double*>(label), label_dim, label_shape,
                                            reinterpret_cast<double*>(loss_val), stream);
      break;
    default:
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_train_multiarg(const char* name, torchfort_tensor_list_t inputs, torchfort_tensor_list_t labels,
                                            float* loss_val, torchfort_tensor_list_t extra_loss_args, cudaStream_t stream) {
  using namespace torchfort;
  try {
    torchfort::train_multiarg(name, inputs, labels, loss_val, extra_loss_args, stream);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_inference(const char* name, void* input, size_t input_dim, int64_t* input_shape,
                                       void* output, size_t output_dim, int64_t* output_shape,
                                       torchfort_datatype_t dtype, cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      torchfort::inference<torchfort::RowMajor>(name, reinterpret_cast<float*>(input), input_dim, input_shape,
                                                reinterpret_cast<float*>(output), output_dim, output_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      torchfort::inference<torchfort::RowMajor>(name, reinterpret_cast<double*>(input), input_dim, input_shape,
                                                reinterpret_cast<double*>(output), output_dim, output_shape, stream);
      break;
    default:
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_inference_F(const char* name, void* input, size_t input_dim, int64_t* input_shape,
                                         void* output, size_t output_dim, int64_t* output_shape,
                                         torchfort_datatype_t dtype, cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      torchfort::inference<torchfort::ColMajor>(name, reinterpret_cast<float*>(input), input_dim, input_shape,
                                                reinterpret_cast<float*>(output), output_dim, output_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      torchfort::inference<torchfort::ColMajor>(name, reinterpret_cast<double*>(input), input_dim, input_shape,
                                                reinterpret_cast<double*>(output), output_dim, output_shape, stream);
      break;
    default:
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_inference_multiarg(const char* name, torchfort_tensor_list_t inputs,
                                                torchfort_tensor_list_t outputs, cudaStream_t stream) {
  using namespace torchfort;
  try {
    torchfort::inference_multiarg(name, inputs, outputs, stream);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_save_model(const char* name, const char* fname) {
  using namespace torchfort;
  try {
    if (models.count(name) == 0) {
      THROW_INVALID_USAGE("Invalid model name provided.");
    }
    models[name].model->save(fname);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_load_model(const char* name, const char* fname) {
  using namespace torchfort;
  try {
    if (models.count(name) == 0) {
      THROW_INVALID_USAGE("Invalid model name provided.");
    }
    models[name].model->load(fname);
    if (models[name].optimizer) {
      models[name].optimizer->parameters() = models[name].model->parameters();
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_save_checkpoint(const char* name, const char* checkpoint_dir) {
  using namespace torchfort;
  try {
    std::filesystem::path root_dir{checkpoint_dir};

    if (!std::filesystem::exists(root_dir)) {
      bool rv = std::filesystem::create_directory(root_dir);
      if (!rv) {
        THROW_INVALID_USAGE("Could not create checkpoint directory " + root_dir.native() + ".");
      }
    }

    if (models.count(name) == 0) {
      THROW_INVALID_USAGE("Invalid model name provided.");
    }
    save_model_pack(models[name], root_dir, true);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_load_checkpoint(const char* name, const char* checkpoint_dir, int64_t* step_train,
                                             int64_t* step_inference) {
  using namespace torchfort;
  try {
    std::filesystem::path root_dir{checkpoint_dir};

    if (models.count(name) == 0) {
      THROW_INVALID_USAGE("Invalid model name provided.");
    }
    load_model_pack(models[name], root_dir, true);

    if (step_train) {
      *step_train = models[name].state->step_train;
    }

    if (step_inference) {
      *step_inference = models[name].state->step_inference;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_tensor_list_create(torchfort_tensor_list_t* tensor_list) {
  *tensor_list = new torchfort::TensorList;
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_tensor_list_destroy(torchfort_tensor_list_t tensor_list_in) {
  auto tensor_list = static_cast<torchfort::TensorList*>(tensor_list_in);
  delete tensor_list;
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_tensor_list_add_tensor(torchfort_tensor_list_t tensor_list_in, void* data_ptr, size_t dim,
                                                    int64_t* shape, torchfort_datatype_t dtype) {
  using namespace torchfort;
  try {
    auto tensor_list = static_cast<torchfort::TensorList*>(tensor_list_in);
    switch (dtype) {
    case TORCHFORT_FLOAT:
      tensor_list->add_tensor<torchfort::RowMajor>(reinterpret_cast<float*>(data_ptr), dim, shape);
      break;
    case TORCHFORT_DOUBLE:
      tensor_list->add_tensor<torchfort::RowMajor>(reinterpret_cast<double*>(data_ptr), dim, shape);
      break;
    case TORCHFORT_INT32:
      tensor_list->add_tensor<torchfort::RowMajor>(reinterpret_cast<int32_t*>(data_ptr), dim, shape);
      break;
    case TORCHFORT_INT64:
      tensor_list->add_tensor<torchfort::RowMajor>(reinterpret_cast<int64_t*>(data_ptr), dim, shape);
      break;
    default:
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_tensor_list_add_tensor_F(torchfort_tensor_list_t tensor_list_in, void* data_ptr, size_t dim,
                                                      int64_t* shape, torchfort_datatype_t dtype) {
  using namespace torchfort;
  try {
    auto tensor_list = static_cast<torchfort::TensorList*>(tensor_list_in);
    switch (dtype) {
    case TORCHFORT_FLOAT:
      tensor_list->add_tensor<torchfort::ColMajor>(reinterpret_cast<float*>(data_ptr), dim, shape);
      break;
    case TORCHFORT_DOUBLE:
      tensor_list->add_tensor<torchfort::ColMajor>(reinterpret_cast<double*>(data_ptr), dim, shape);
      break;
    case TORCHFORT_INT32:
      tensor_list->add_tensor<torchfort::ColMajor>(reinterpret_cast<int32_t*>(data_ptr), dim, shape);
      break;
    case TORCHFORT_INT64:
      tensor_list->add_tensor<torchfort::ColMajor>(reinterpret_cast<int64_t*>(data_ptr), dim, shape);
      break;
    default:
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}
