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
#include "internal/rl/off_policy/ddpg.h"
#include "internal/rl/off_policy/sac.h"
#include "internal/rl/off_policy/td3.h"

namespace torchfort {
namespace rl {

namespace off_policy {

std::unordered_map<std::string, std::shared_ptr<RLOffPolicySystem>> registry;

// default constructor:
RLOffPolicySystem::RLOffPolicySystem(int model_device, int rb_device)
    : train_step_count_(0), model_device_(get_device(model_device)), rb_device_(get_device(rb_device)) {
  if (!(torchfort::rl::validate_devices(model_device, rb_device))) {
    THROW_INVALID_USAGE("The parameters model_device and rb_device have to specify the same GPU or one has to specify "
                        "a GPU and the other the CPU.");
  }
}

} // namespace off_policy

} // namespace rl
} // namespace torchfort

torchfort_result_t torchfort_rl_off_policy_create_system(const char* name, const char* config_fname, int model_device,
                                                         int rb_device) {
  using namespace torchfort;

  try {
    YAML::Node config = YAML::LoadFile(config_fname);

    // Setting up model
    if (config["algorithm"]) {
      auto algorith_node = config["algorithm"];
      if (algorith_node["type"]) {
        auto algorithm_type = sanitize(algorith_node["type"].as<std::string>());

        if (algorithm_type == "td3") {
          rl::off_policy::registry[sanitize(name)] =
              std::make_shared<rl::off_policy::TD3System>(name, config, model_device, rb_device);
        } else if (algorithm_type == "sac") {
          rl::off_policy::registry[sanitize(name)] =
              std::make_shared<rl::off_policy::SACSystem>(name, config, model_device, rb_device);
        } else if (algorithm_type == "ddpg") {
          rl::off_policy::registry[sanitize(name)] =
              std::make_shared<rl::off_policy::DDPGSystem>(name, config, model_device, rb_device);
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

torchfort_result_t torchfort_rl_off_policy_create_distributed_system(const char* name, const char* config_fname,
                                                                     MPI_Comm mpi_comm, int model_device,
                                                                     int rb_device) {
  using namespace torchfort;

  try {
    torchfort_rl_off_policy_create_system(name, config_fname, model_device, rb_device);
    rl::off_policy::registry[sanitize(name)]->initSystemComm(mpi_comm);
    // set the RNG for the rollout buffer
    unsigned int seed = 5489u + static_cast<unsigned int>(rl::off_policy::registry[sanitize(name)]->getRank());
    rl::off_policy::registry[sanitize(name)]->setSeed(seed);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// save system
torchfort_result_t torchfort_rl_off_policy_save_checkpoint(const char* name, const char* checkpoint_dir) {
  using namespace torchfort;

  try {
    rl::off_policy::registry[name]->saveCheckpoint(checkpoint_dir);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// load system
torchfort_result_t torchfort_rl_off_policy_load_checkpoint(const char* name, const char* checkpoint_dir) {
  using namespace torchfort;

  try {
    rl::off_policy::registry[name]->loadCheckpoint(checkpoint_dir);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// ready check
torchfort_result_t torchfort_rl_off_policy_is_ready(const char* name, bool& ready) {
  using namespace torchfort;
  try {
    ready = rl::off_policy::registry[name]->isReady();
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// train step
torchfort_result_t torchfort_rl_off_policy_train_step(const char* name, float* p_loss_val, float* q_loss_val,
                                                      cudaStream_t ext_stream) {
  using namespace torchfort;

  // TODO: we need to figure out what to do if RB and Model streams are different
  auto model_device = rl::off_policy::registry[name]->modelDevice();
  auto rb_device = rl::off_policy::registry[name]->rbDevice();
#ifdef ENABLE_GPU
  c10::cuda::OptionalCUDAStreamGuard guard;
  if (model_device.is_cuda()) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, model_device.index());
    guard.reset_stream(stream);
  } else if (rb_device.is_cuda()) {
    auto stream = c10::cuda::getStreamFromExternal(ext_stream, rb_device.index());
    guard.reset_stream(stream);
  }
#endif

  try {
    // perform a training step
    rl::off_policy::registry[name]->trainStep(*p_loss_val, *q_loss_val);
  } catch (const BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return TORCHFORT_RESULT_SUCCESS;
}

// LOGGING
RL_OFF_POLICY_WANDB_LOG_FUNC(int)
RL_OFF_POLICY_WANDB_LOG_FUNC(float)
RL_OFF_POLICY_WANDB_LOG_FUNC(double)

// RB utilities
torchfort_result_t torchfort_rl_off_policy_update_replay_buffer(const char* name, void* state_old, void* state_new,
                                                                size_t state_dim, int64_t* state_shape,
                                                                void* action_old, size_t action_dim,
                                                                int64_t* action_shape, const void* reward,
                                                                bool final_state, torchfort_datatype_t dtype,
                                                                cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      float reward_val = *reinterpret_cast<const float*>(reward);
      rl::off_policy::update_replay_buffer<RowMajor>(
          name, reinterpret_cast<float*>(state_old), reinterpret_cast<float*>(state_new), state_dim, state_shape,
          reinterpret_cast<float*>(action_old), action_dim, action_shape, reward_val, final_state, ext_stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      double reward_val = *reinterpret_cast<const double*>(reward);
      rl::off_policy::update_replay_buffer<RowMajor>(
          name, reinterpret_cast<double*>(state_old), reinterpret_cast<double*>(state_new), state_dim, state_shape,
          reinterpret_cast<double*>(action_old), action_dim, action_shape, reward_val, final_state, ext_stream);
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

torchfort_result_t torchfort_rl_off_policy_update_replay_buffer_F(const char* name, void* state_old, void* state_new,
                                                                  size_t state_dim, int64_t* state_shape,
                                                                  void* action_old, size_t action_dim,
                                                                  int64_t* action_shape, const void* reward,
                                                                  bool final_state, torchfort_datatype_t dtype,
                                                                  cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      float reward_val = *reinterpret_cast<const float*>(reward);
      rl::off_policy::update_replay_buffer<ColMajor>(
          name, reinterpret_cast<float*>(state_old), reinterpret_cast<float*>(state_new), state_dim, state_shape,
          reinterpret_cast<float*>(action_old), action_dim, action_shape, reward_val, final_state, stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      double reward_val = *reinterpret_cast<const double*>(reward);
      rl::off_policy::update_replay_buffer<ColMajor>(
          name, reinterpret_cast<double*>(state_old), reinterpret_cast<double*>(state_new), state_dim, state_shape,
          reinterpret_cast<double*>(action_old), action_dim, action_shape, reward_val, final_state, stream);
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

torchfort_result_t torchfort_rl_off_policy_update_replay_buffer_multi(const char* name, void* state_old,
								      void* state_new, size_t state_dim, int64_t* state_shape,
                                                                      void* action_old, size_t action_dim, int64_t* action_shape,
								      void* reward, size_t reward_dim, int64_t* reward_shape,
                                                                      void* final_state, size_t final_state_dim, int64_t* final_state_shape,
								      torchfort_datatype_t dtype, cudaStream_t ext_stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
      rl::off_policy::update_replay_buffer<RowMajor>(
          name, reinterpret_cast<float*>(state_old),
	  reinterpret_cast<float*>(state_new), state_dim, state_shape,
          reinterpret_cast<float*>(action_old), action_dim, action_shape,
	  reinterpret_cast<float*>(reward), reward_dim, reward_shape,
	  reinterpret_cast<float*>(final_state), final_state_dim, final_state_shape,
	  ext_stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      rl::off_policy::update_replay_buffer<RowMajor>(
          name, reinterpret_cast<double*>(state_old),
	  reinterpret_cast<double*>(state_new), state_dim, state_shape,
          reinterpret_cast<double*>(action_old), action_dim, action_shape,
	  reinterpret_cast<double*>(reward), reward_dim, reward_shape,
          reinterpret_cast<double*>(final_state), final_state_dim, final_state_shape,
	  ext_stream);
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

torchfort_result_t torchfort_rl_off_policy_update_replay_buffer_multi_F(const char* name, void* state_old,
									void* state_new, size_t state_dim, int64_t* state_shape,
									void* action_old, size_t action_dim, int64_t* action_shape,
									void* reward, size_t reward_dim, int64_t* reward_shape,
									void* final_state, size_t final_state_dim, int64_t* final_state_shape,
									torchfort_datatype_t dtype, cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT: {
     
      rl::off_policy::update_replay_buffer<ColMajor>(
						     name, reinterpret_cast<float*>(state_old),
						     reinterpret_cast<float*>(state_new), state_dim, state_shape,
						     reinterpret_cast<float*>(action_old), action_dim, action_shape,
						     reinterpret_cast<float*>(reward), reward_dim, reward_shape,
						     reinterpret_cast<float*>(final_state), final_state_dim, final_state_shape,
						     stream);
      break;
    }
    case TORCHFORT_DOUBLE: {
      rl::off_policy::update_replay_buffer<ColMajor>(
						     name, reinterpret_cast<double*>(state_old),
						     reinterpret_cast<double*>(state_new), state_dim, state_shape,
						     reinterpret_cast<double*>(action_old), action_dim, action_shape,
						     reinterpret_cast<double*>(reward), reward_dim, reward_shape,
                                                     reinterpret_cast<double*>(final_state), final_state_dim, final_state_shape,
						     stream);
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


torchfort_result_t torchfort_rl_off_policy_predict_explore(const char* name, void* state, size_t state_dim,
                                                           int64_t* state_shape, void* action, size_t action_dim,
                                                           int64_t* action_shape, torchfort_datatype_t dtype,
                                                           cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::off_policy::predict_explore<torchfort::RowMajor>(name, reinterpret_cast<float*>(state), state_dim,
                                                           state_shape, reinterpret_cast<float*>(action), action_dim,
                                                           action_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::off_policy::predict_explore<torchfort::RowMajor>(name, reinterpret_cast<double*>(state), state_dim,
                                                           state_shape, reinterpret_cast<double*>(action), action_dim,
                                                           action_shape, stream);
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

torchfort_result_t torchfort_rl_off_policy_predict_explore_F(const char* name, void* state, size_t state_dim,
                                                             int64_t* state_shape, void* action, size_t action_dim,
                                                             int64_t* action_shape, torchfort_datatype_t dtype,
                                                             cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::off_policy::predict_explore<torchfort::ColMajor>(name, reinterpret_cast<float*>(state), state_dim,
                                                           state_shape, reinterpret_cast<float*>(action), action_dim,
                                                           action_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::off_policy::predict_explore<torchfort::ColMajor>(name, reinterpret_cast<double*>(state), state_dim,
                                                           state_shape, reinterpret_cast<double*>(action), action_dim,
                                                           action_shape, stream);
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

torchfort_result_t torchfort_rl_off_policy_predict(const char* name, void* state, size_t state_dim,
                                                   int64_t* state_shape, void* action, size_t action_dim,
                                                   int64_t* action_shape, torchfort_datatype_t dtype,
                                                   cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::off_policy::predict<RowMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                        reinterpret_cast<float*>(action), action_dim, action_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::off_policy::predict<RowMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                        reinterpret_cast<double*>(action), action_dim, action_shape, stream);
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

torchfort_result_t torchfort_rl_off_policy_predict_F(const char* name, void* state, size_t state_dim,
                                                     int64_t* state_shape, void* action, size_t action_dim,
                                                     int64_t* action_shape, torchfort_datatype_t dtype,
                                                     cudaStream_t stream) {
  using namespace torchfort;
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::off_policy::predict<ColMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                        reinterpret_cast<float*>(action), action_dim, action_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::off_policy::predict<ColMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                        reinterpret_cast<double*>(action), action_dim, action_shape, stream);
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

torchfort_result_t torchfort_rl_off_policy_evaluate(const char* name, void* state, size_t state_dim,
                                                    int64_t* state_shape, void* action, size_t action_dim,
                                                    int64_t* action_shape, void* reward, size_t reward_dim,
                                                    int64_t* reward_shape, torchfort_datatype_t dtype,
                                                    cudaStream_t stream) {
  using namespace torchfort;

  if (reward_dim != 1) {
    THROW_INVALID_USAGE("The dimension of the reward array has to be equal to 1.");
  }
  
  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::off_policy::policy_evaluate<RowMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                                reinterpret_cast<float*>(action), action_dim, action_shape,
                                                reinterpret_cast<float*>(reward), reward_dim, reward_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::off_policy::policy_evaluate<RowMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                                reinterpret_cast<double*>(action), action_dim, action_shape,
                                                reinterpret_cast<double*>(reward), reward_dim, reward_shape, stream);
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

torchfort_result_t torchfort_rl_off_policy_evaluate_F(const char* name, void* state, size_t state_dim,
                                                      int64_t* state_shape, void* action, size_t action_dim,
                                                      int64_t* action_shape, void* reward, size_t reward_dim,
                                                      int64_t* reward_shape, torchfort_datatype_t dtype,
                                                      cudaStream_t stream) {
  using namespace torchfort;

  if (reward_dim != 1) {
    THROW_INVALID_USAGE("The dimension of the reward array has to be equal to 1.");
  }

  try {
    switch (dtype) {
    case TORCHFORT_FLOAT:
      rl::off_policy::policy_evaluate<ColMajor>(name, reinterpret_cast<float*>(state), state_dim, state_shape,
                                                reinterpret_cast<float*>(action), action_dim, action_shape,
                                                reinterpret_cast<float*>(reward), reward_dim, reward_shape, stream);
      break;
    case TORCHFORT_DOUBLE:
      rl::off_policy::policy_evaluate<ColMajor>(name, reinterpret_cast<double*>(state), state_dim, state_shape,
                                                reinterpret_cast<double*>(action), action_dim, action_shape,
                                                reinterpret_cast<double*>(reward), reward_dim, reward_shape, stream);
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
