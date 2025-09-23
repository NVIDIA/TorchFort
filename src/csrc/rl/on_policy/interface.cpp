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
#include "internal/model_wrapper.h"
#include "internal/models.h"
#include "internal/rl/rl.h"
#include "internal/setup.h"
#include "internal/utils.h"
#include "torchfort.h"

// special stuff
#include "internal/rl/on_policy/ppo.h"

namespace torchfort {
namespace rl {

namespace on_policy {
// Global variables
std::unordered_map<std::string, std::shared_ptr<RLOnPolicySystem>> registry;

// default constructor:
RLOnPolicySystem::RLOnPolicySystem(int model_device, int rb_device)
    : train_step_count_(0), model_device_(get_device(model_device)), rb_device_(get_device(rb_device)) {
  if (!(torchfort::rl::validate_devices(model_device, rb_device))) {
    THROW_INVALID_USAGE("The parameters model_device and rb_device have to specify the same GPU or one has to specify "
                        "a GPU and the other the CPU.");
  }
}

} // namespace on_policy

} // namespace rl
} // namespace torchfort

torchfort_result_t torchfort_rl_on_policy_create_system(const char* name, const char* config_fname, int model_device,
                                                        int rb_device) {
  using namespace torchfort;

  try {
    YAML::Node config = YAML::LoadFile(config_fname);

    // Setting up model
    if (config["algorithm"]) {
      auto algorith_node = config["algorithm"];
      if (algorith_node["type"]) {
        auto algorithm_type = sanitize(algorith_node["type"].as<std::string>());

        if (algorithm_type == "ppo") {
          rl::on_policy::registry[sanitize(name)] =
              std::make_shared<rl::on_policy::PPOSystem>(name, config, model_device, rb_device);
        } else {
          THROW_INVALID_USAGE(algorithm_type);
        }
      } else {
        THROW_INVALID_USAGE("Missing type specifier in algorithm block in configuration file.");
      }
    } else {
      THROW_INVALID_USAGE("Missing algorithm block in configuration file.");
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_rl_on_policy_create_distributed_system(const char* name, const char* config_fname,
                                                                    MPI_Comm mpi_comm, int model_device,
                                                                    int rb_device) {
  using namespace torchfort;

  try {
    torchfort_rl_on_policy_create_system(name, config_fname, model_device, rb_device);
    rl::on_policy::registry[sanitize(name)]->initSystemComm(mpi_comm);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// save system
torchfort_result_t torchfort_rl_on_policy_save_checkpoint(const char* name, const char* checkpoint_dir) {
  using namespace torchfort;

  try {
    rl::on_policy::registry[name]->saveCheckpoint(checkpoint_dir);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// load system
torchfort_result_t torchfort_rl_on_policy_load_checkpoint(const char* name, const char* checkpoint_dir) {
  using namespace torchfort;

  try {
    rl::on_policy::registry[name]->loadCheckpoint(checkpoint_dir);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// ready check
torchfort_result_t torchfort_rl_on_policy_is_ready(const char* name, bool& ready) {
  using namespace torchfort;
  try {
    ready = rl::on_policy::registry[name]->isReady();
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// train step
torchfort_result_t torchfort_rl_on_policy_train_step(const char* name, float* p_loss_val, float* q_loss_val,
                                                     cudaStream_t ext_stream) {
  using namespace torchfort;

#ifdef ENABLE_GPU
  // TODO: we need to figure out what to do if RB and Model streams are different
  c10::cuda::OptionalCUDAStreamGuard guard;
  auto model_device = rl::on_policy::registry[name]->modelDevice();
  if (model_device.is_cuda()) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, model_device.index());
    guard.reset_stream(stream);
  }
#endif

  try {
    // perform a training step
    rl::on_policy::registry[name]->trainStep(*p_loss_val, *q_loss_val);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// LOGGING
RL_ON_POLICY_WANDB_LOG_FUNC(int)
RL_ON_POLICY_WANDB_LOG_FUNC(float)
RL_ON_POLICY_WANDB_LOG_FUNC(double)

// RB utilities
// single env convenience routine
torchfort_result_t torchfort_rl_on_policy_update_rollout_buffer(const char* name, void* state, size_t state_dim,
                                                                int64_t* state_shape, void* action, size_t action_dim,
                                                                int64_t* action_shape, const void* reward,
                                                                bool final_state, torchfort_datatype_t dtype,
                                                                cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      float reward_val = *reinterpret_cast<const float*>(reward);
      rl::on_policy::update_rollout_buffer<RowMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                                     reinterpret_cast<float*>(action), action_dim, action_shape,
                                                     reward_val, final_state, ext_stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      double reward_val = *reinterpret_cast<const double*>(reward);
      rl::on_policy::update_rollout_buffer<RowMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                                     reinterpret_cast<double*>(action), action_dim, action_shape,
                                                     reward_val, final_state, ext_stream);
      break;
    }
    default: {
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_rl_on_policy_update_rollout_buffer_F(const char* name, void* state, size_t state_dim,
                                                                  int64_t* state_shape, void* action, size_t action_dim,
                                                                  int64_t* action_shape, const void* reward,
                                                                  bool final_state, torchfort_datatype_t dtype,
                                                                  cudaStream_t ext_stream) {

  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      float reward_val = *reinterpret_cast<const float*>(reward);
      rl::on_policy::update_rollout_buffer<ColMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                                     reinterpret_cast<float*>(action), action_dim, action_shape,
                                                     reward_val, final_state, ext_stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      double reward_val = *reinterpret_cast<const double*>(reward);
      rl::on_policy::update_rollout_buffer<ColMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                                     reinterpret_cast<double*>(action), action_dim, action_shape,
                                                     reward_val, final_state, ext_stream);
      break;
    }
    default: {
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// multi env
torchfort_result_t torchfort_rl_on_policy_update_rollout_buffer_multi(
    const char* name, void* state, size_t state_dim, int64_t* state_shape, void* action, size_t action_dim,
    int64_t* action_shape, void* reward, size_t reward_dim, int64_t* reward_shape, void* final_state,
    size_t final_state_dim, int64_t* final_state_shape, torchfort_datatype_t dtype, cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      rl::on_policy::update_rollout_buffer<RowMajor>(
          name, reinterpret_cast<float*>(state), state_dim, state_shape, reinterpret_cast<float*>(action), action_dim,
          action_shape, reinterpret_cast<float*>(reward), reward_dim, reward_shape,
          reinterpret_cast<float*>(final_state), final_state_dim, final_state_shape, ext_stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      rl::on_policy::update_rollout_buffer<RowMajor>(
          name, reinterpret_cast<double*>(state), state_dim, state_shape, reinterpret_cast<double*>(action), action_dim,
          action_shape, reinterpret_cast<double*>(reward), reward_dim, reward_shape,
          reinterpret_cast<double*>(final_state), final_state_dim, final_state_shape, ext_stream);
      break;
    }
    default: {
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_rl_on_policy_update_rollout_buffer_multi_F(
    const char* name, void* state, size_t state_dim, int64_t* state_shape, void* action, size_t action_dim,
    int64_t* action_shape, void* reward, size_t reward_dim, int64_t* reward_shape, void* final_state,
    size_t final_state_dim, int64_t* final_state_shape, torchfort_datatype_t dtype, cudaStream_t ext_stream) {

  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      rl::on_policy::update_rollout_buffer<ColMajor>(
          name, reinterpret_cast<float*>(state), state_dim, state_shape, reinterpret_cast<float*>(action), action_dim,
          action_shape, reinterpret_cast<float*>(reward), reward_dim, reward_shape,
          reinterpret_cast<float*>(final_state), final_state_dim, final_state_shape, ext_stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      rl::on_policy::update_rollout_buffer<ColMajor>(
          name, reinterpret_cast<double*>(state), state_dim, state_shape, reinterpret_cast<double*>(action), action_dim,
          action_shape, reinterpret_cast<double*>(reward), reward_dim, reward_shape,
          reinterpret_cast<double*>(final_state), final_state_dim, final_state_shape, ext_stream);
      break;
    }
    default: {
      THROW_INVALID_USAGE("Unknown datatype provided.");
      break;
    }
    }
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_rl_on_policy_reset_rollout_buffer(const char* name) {
  using namespace torchfort;
  try {
    rl::on_policy::registry[name]->resetRolloutBuffer();
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

torchfort_result_t torchfort_rl_on_policy_predict_explore(const char* name, void* state, size_t state_dim,
                                                          int64_t* state_shape, void* action, size_t action_dim,
                                                          int64_t* action_shape, torchfort_datatype_t dtype,
                                                          cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::on_policy::predict_explore<torchfort::RowMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                                          reinterpret_cast<float*>(action), action_dim, action_shape,
                                                          ext_stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::on_policy::predict_explore<torchfort::RowMajor>(name, reinterpret_cast<double*>(state), state_dim,
                                                          state_shape, reinterpret_cast<double*>(action), action_dim,
                                                          action_shape, ext_stream);
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

torchfort_result_t torchfort_rl_on_policy_predict_explore_F(const char* name, void* state, size_t state_dim,
                                                            int64_t* state_shape, void* action, size_t action_dim,
                                                            int64_t* action_shape, torchfort_datatype_t dtype,
                                                            cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::on_policy::predict_explore<torchfort::ColMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                                          reinterpret_cast<float*>(action), action_dim, action_shape,
                                                          ext_stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::on_policy::predict_explore<torchfort::ColMajor>(name, reinterpret_cast<double*>(state), state_dim,
                                                          state_shape, reinterpret_cast<double*>(action), action_dim,
                                                          action_shape, ext_stream);
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

torchfort_result_t torchfort_rl_on_policy_predict(const char* name, void* state, size_t state_dim, int64_t* state_shape,
                                                  void* action, size_t action_dim, int64_t* action_shape,
                                                  torchfort_datatype_t dtype, cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::on_policy::predict<RowMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                       reinterpret_cast<float*>(action), action_dim, action_shape, ext_stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::on_policy::predict<RowMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                       reinterpret_cast<double*>(action), action_dim, action_shape, ext_stream);
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

torchfort_result_t torchfort_rl_on_policy_predict_F(const char* name, void* state, size_t state_dim,
                                                    int64_t* state_shape, void* action, size_t action_dim,
                                                    int64_t* action_shape, torchfort_datatype_t dtype,
                                                    cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::on_policy::predict<ColMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                       reinterpret_cast<float*>(action), action_dim, action_shape, ext_stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::on_policy::predict<ColMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                       reinterpret_cast<double*>(action), action_dim, action_shape, ext_stream);
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

torchfort_result_t torchfort_rl_on_policy_evaluate(const char* name, void* state, size_t state_dim,
                                                   int64_t* state_shape, void* action, size_t action_dim,
                                                   int64_t* action_shape, void* reward, size_t reward_dim,
                                                   int64_t* reward_shape, torchfort_datatype_t dtype,
                                                   cudaStream_t ext_stream) {
  using namespace torchfort;

  if (reward_dim != 1) {
    THROW_INVALID_USAGE("The dimension of the reward array has to be equal to 1.");
  }

  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::on_policy::policy_evaluate<RowMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                               reinterpret_cast<float*>(action), action_dim, action_shape,
                                               reinterpret_cast<float*>(reward), reward_dim, reward_shape, ext_stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::on_policy::policy_evaluate<RowMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                               reinterpret_cast<double*>(action), action_dim, action_shape,
                                               reinterpret_cast<double*>(reward), reward_dim, reward_shape, ext_stream);
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

torchfort_result_t torchfort_rl_on_policy_evaluate_F(const char* name, void* state, size_t state_dim,
                                                     int64_t* state_shape, void* action, size_t action_dim,
                                                     int64_t* action_shape, void* reward, size_t reward_dim,
                                                     int64_t* reward_shape, torchfort_datatype_t dtype,
                                                     cudaStream_t ext_stream) {
  using namespace torchfort;

  if (reward_dim != 1) {
    THROW_INVALID_USAGE("The dimension of the reward array has to be equal to 1.");
  }

  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::on_policy::policy_evaluate<ColMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                               reinterpret_cast<float*>(action), action_dim, action_shape,
                                               reinterpret_cast<float*>(reward), reward_dim, reward_shape, ext_stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::on_policy::policy_evaluate<ColMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                               reinterpret_cast<double*>(action), action_dim, action_shape,
                                               reinterpret_cast<double*>(reward), reward_dim, reward_shape, ext_stream);
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
