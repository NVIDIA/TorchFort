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

#include <gtest/gtest.h>

#include "torchfort.h"
#include "internal/defines.h"
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <random>

struct CoutRedirect {
  CoutRedirect(std::streambuf* new_buffer) : old(std::cout.rdbuf(new_buffer)) {}
  
  ~CoutRedirect() { std::cout.rdbuf(old); }
  
  private:
    std::streambuf* old;
};

// Function to modify TD3 config and write to temporary file
std::string createModifiedTD3Config(int state_size, int action_size) {
  // Read the original TD3 config
  std::string config_path = "configs/td3.yaml";
  YAML::Node config = YAML::LoadFile(config_path);
  
  // Modify the policy model layer sizes
  if (config["policy_model"] && config["policy_model"]["parameters"] && config["policy_model"]["parameters"]["layer_sizes"]) {
    auto layer_sizes = config["policy_model"]["parameters"]["layer_sizes"];
    
    // Modify first layer (input) to match state_size
    layer_sizes[0] = state_size;
    
    // Modify last layer (output) to match action_size
    if (layer_sizes.size() > 0) {
      layer_sizes[layer_sizes.size() - 1] = action_size;
    }
  }
  
  // Also modify critic model to account for state + action input
  if (config["critic_model"] && config["critic_model"]["parameters"] && config["critic_model"]["parameters"]["layer_sizes"]) {
    auto critic_layer_sizes = config["critic_model"]["parameters"]["layer_sizes"];
    
    // Critic takes state + action as input
    critic_layer_sizes[0] = state_size + action_size;
  }
  
  // Create temporary file
  std::string temp_filename("./tmpconfig.yaml");
  
  // Write modified config to temporary file
  std::ofstream temp_file(temp_filename);
  temp_file << config;
  temp_file.close();
  
  return temp_filename;
}

// Function to modify PPO config and write to temporary file
std::string createModifiedPPOConfig(int state_size, int action_size) {
  // Read the original TD3 config
  std::string config_path = "configs/ppo.yaml";
  YAML::Node config = YAML::LoadFile(config_path);

  // Modify the policy model layer sizes
  if (config["actor_critic_model"] && config["actor_critic_model"]["parameters"]) { 
    if (config["actor_critic_model"]["parameters"]["encoder_layer_sizes"]) {
      auto layer_sizes = config["actor_critic_model"]["parameters"]["encoder_layer_sizes"];

      // Modify first layer (input) to match state_size
      layer_sizes[0] = state_size;

      // Modify last layer (output) to match action_size
      if (layer_sizes.size() > 0) {
        layer_sizes[layer_sizes.size() - 1] = action_size;
      }
    }
    if (config["actor_critic_model"]["parameters"]["actor_layer_sizes"]) {
      auto layer_sizes = config["actor_critic_model"]["parameters"]["actor_layer_sizes"];
      layer_sizes[layer_sizes.size() - 1] = action_size;
    }
  }

  // Create temporary file
  std::string temp_filename("./tmpconfig.yaml");

  // Write modified config to temporary file
  std::ofstream temp_file(temp_filename);
  temp_file << config;
  temp_file.close();

  return temp_filename;
}

class TorchFortInterface : public testing::TestWithParam<std::vector<std::vector<int64_t>>> {};

// it is only necessary to test the interface for TD3 since DDPG has the same interface
TEST_P(TorchFortInterface, DDPGTD3Predict) {

  // capture stdout
  std::stringstream buffer;

  std::shared_ptr<CoutRedirect> red;
  red = std::make_shared<CoutRedirect>(buffer.rdbuf());

  // set rng
  torchfort_set_manual_seed(333);

  auto shapes = GetParam();
  std::vector<int64_t> state_shape = shapes[0];
  std::vector<int64_t> action_shape = shapes[1];

  int64_t state_size, batch_size;
  if (state_shape.size() == 1) {
    batch_size = 1;
    state_size = state_shape[0];
  } else {
    batch_size = state_shape[0];
    state_size = state_shape[1];
  }
  int64_t action_size = action_shape[action_shape.size() - 1];

  // Create modified TD3 config with correct state size
  std::string temp_config_file = createModifiedTD3Config(state_size, action_size);
  
  // allocate state
  float* state = new float[state_size*batch_size];
  float* action = new float[action_size*batch_size];
  
  // Initialize state and action with random values using modern C++ RNG
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  
  for (int64_t i = 0; i < state_size * batch_size; ++i) {
    state[i] = dis(gen); // Random values in [-1, 1]
  }
  
  // instantiate RL system
  torchfort_rl_off_policy_create_system("test", temp_config_file.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);

  // try a prediction
  torchfort_rl_off_policy_predict("test", state, state_shape.size(), state_shape.data(), action, action_shape.size(), action_shape.data(), TORCHFORT_FLOAT, 0);
  
  // Clean up
  delete[] state;
  delete[] action;
  std::filesystem::remove(temp_config_file);
}

// it is only necessary to test the interface for TD3 since DDPG has the same interface
TEST_P(TorchFortInterface, DDPGTD3PredictExplore) {

    // capture stdout
    std::stringstream buffer;
  
    std::shared_ptr<CoutRedirect> red;
    red = std::make_shared<CoutRedirect>(buffer.rdbuf());
  
    // set rng
    torchfort_set_manual_seed(333);
  
    auto shapes = GetParam();
    std::vector<int64_t> state_shape = shapes[0];
    std::vector<int64_t> action_shape = shapes[1];
  
    int64_t state_size, batch_size;
    if (state_shape.size() == 1) {
      batch_size = 1;
      state_size = state_shape[0];
    } else {
      batch_size = state_shape[0];
      state_size = state_shape[1];
    }
    int64_t action_size = action_shape[action_shape.size() - 1];
  
    // Create modified TD3 config with correct state size
    std::string temp_config_file = createModifiedTD3Config(state_size, action_size);
    
    // allocate state
    float* state = new float[state_size*batch_size];
    float* action = new float[action_size*batch_size];
    
    // Initialize state and action with random values using modern C++ RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int64_t i = 0; i < state_size * batch_size; ++i) {
      state[i] = dis(gen); // Random values in [-1, 1]
    }
    
    // instantiate RL system
    torchfort_rl_off_policy_create_system("test", temp_config_file.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  
    // try a prediction with noise (explore)
    torchfort_rl_off_policy_predict_explore("test", state, state_shape.size(), state_shape.data(), action, action_shape.size(), action_shape.data(), TORCHFORT_FLOAT, 0);
    
    // Clean up
    delete[] state;
    delete[] action;
    std::filesystem::remove(temp_config_file);
}

TEST_P(TorchFortInterface, DDPGTD3Evaluate) {

    // capture stdout
    std::stringstream buffer;
  
    std::shared_ptr<CoutRedirect> red;
    red = std::make_shared<CoutRedirect>(buffer.rdbuf());
  
    // set rng
    torchfort_set_manual_seed(333);
  
    auto shapes = GetParam();
    std::vector<int64_t> state_shape = shapes[0];
    std::vector<int64_t> action_shape = shapes[1];
  
    int64_t state_size, batch_size;
    if (state_shape.size() == 1) {
      batch_size = 1;
      state_size = state_shape[0];
    } else {
      batch_size = state_shape[0];
      state_size = state_shape[1];
    }
    int64_t action_size = action_shape[action_shape.size() - 1];
    std::vector<int64_t> reward_shape = {batch_size};
  
    // Create modified TD3 config with correct state size
    std::string temp_config_file = createModifiedTD3Config(state_size, action_size);
    
    // allocate state
    float* state = new float[state_size*batch_size];
    float* action = new float[action_size*batch_size];
    float* reward = new float[batch_size];
    
    // Initialize state and action with random values using modern C++ RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int64_t i = 0; i < state_size * batch_size; ++i) {
      state[i] = dis(gen); // Random values in [-1, 1]
    }
    
    // instantiate RL system
    torchfort_rl_off_policy_create_system("test", temp_config_file.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  
    // try a prediction with noise (explore)
    torchfort_rl_off_policy_evaluate("test", state, state_shape.size(), state_shape.data(), action, action_shape.size(), action_shape.data(), reward, 1, reward_shape.data(), TORCHFORT_FLOAT, 0);
    
    // Clean up
    delete[] state;
    delete[] action;
    delete[] reward;
    std::filesystem::remove(temp_config_file);
}

TEST_P(TorchFortInterface, PPOPredict) {

    // capture stdout
    std::stringstream buffer;
  
    std::shared_ptr<CoutRedirect> red;
    red = std::make_shared<CoutRedirect>(buffer.rdbuf());
  
    // set rng
    torchfort_set_manual_seed(333);
  
    auto shapes = GetParam();
    std::vector<int64_t> state_shape = shapes[0];
    std::vector<int64_t> action_shape = shapes[1];
  
    int64_t state_size, batch_size;
    if (state_shape.size() == 1) {
      batch_size = 1;
      state_size = state_shape[0];
    } else {
      batch_size = state_shape[0];
      state_size = state_shape[1];
    }
    int64_t action_size = action_shape[action_shape.size() - 1];
  
    // Create modified TD3 config with correct state size
    std::string temp_config_file = createModifiedPPOConfig(state_size, action_size);
    
    // allocate state
    float* state = new float[state_size*batch_size];
    float* action = new float[action_size*batch_size];
    
    // Initialize state and action with random values using modern C++ RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int64_t i = 0; i < state_size * batch_size; ++i) {
      state[i] = dis(gen); // Random values in [-1, 1]
    }
    
    // instantiate RL system
    torchfort_rl_on_policy_create_system("test", temp_config_file.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  
    // try a prediction
    torchfort_rl_on_policy_predict("test", state, state_shape.size(), state_shape.data(), action, action_shape.size(), action_shape.data(), TORCHFORT_FLOAT, 0);
    
    // Clean up
    delete[] state;
    delete[] action;
    std::filesystem::remove(temp_config_file);
}

TEST_P(TorchFortInterface, PPOPredictExplore) {

    // capture stdout
    std::stringstream buffer;
  
    std::shared_ptr<CoutRedirect> red;
    red = std::make_shared<CoutRedirect>(buffer.rdbuf());
  
    // set rng
    torchfort_set_manual_seed(333);
  
    auto shapes = GetParam();
    std::vector<int64_t> state_shape = shapes[0];
    std::vector<int64_t> action_shape = shapes[1];
  
    int64_t state_size, batch_size;
    if (state_shape.size() == 1) {
      batch_size = 1;
      state_size = state_shape[0];
    } else {
      batch_size = state_shape[0];
      state_size = state_shape[1];
    }
    int64_t action_size = action_shape[action_shape.size() - 1];
  
    // Create modified TD3 config with correct state size
    std::string temp_config_file = createModifiedPPOConfig(state_size, action_size);
    
    // allocate state
    float* state = new float[state_size*batch_size];
    float* action = new float[action_size*batch_size];
    
    // Initialize state and action with random values using modern C++ RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int64_t i = 0; i < state_size * batch_size; ++i) {
      state[i] = dis(gen); // Random values in [-1, 1]
    }
    
    // instantiate RL system
    torchfort_rl_on_policy_create_system("test", temp_config_file.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  
    // try a prediction
    torchfort_rl_on_policy_predict_explore("test", state, state_shape.size(), state_shape.data(), action, action_shape.size(), action_shape.data(), TORCHFORT_FLOAT, 0);
    
    // Clean up
    delete[] state;
    delete[] action;
    std::filesystem::remove(temp_config_file);
}

TEST_P(TorchFortInterface, PPOEvaluate) {

    // capture stdout
    std::stringstream buffer;
  
    std::shared_ptr<CoutRedirect> red;
    red = std::make_shared<CoutRedirect>(buffer.rdbuf());
  
    // set rng
    torchfort_set_manual_seed(333);
  
    auto shapes = GetParam();
    std::vector<int64_t> state_shape = shapes[0];
    std::vector<int64_t> action_shape = shapes[1];
  
    int64_t state_size, batch_size;
    if (state_shape.size() == 1) {
      batch_size = 1;
      state_size = state_shape[0];
    } else {
      batch_size = state_shape[0];
      state_size = state_shape[1];
    }
    int64_t action_size = action_shape[action_shape.size() - 1];
    std::vector<int64_t> reward_shape = {batch_size};
  
    // Create modified TD3 config with correct state size
    std::string temp_config_file = createModifiedPPOConfig(state_size, action_size);
    
    // allocate state
    float* state = new float[state_size*batch_size];
    float* action = new float[action_size*batch_size];
    float* reward = new float[batch_size];

    // Initialize state and action with random values using modern C++ RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int64_t i = 0; i < state_size * batch_size; ++i) {
      state[i] = dis(gen); // Random values in [-1, 1]
    }
    
    // instantiate RL system
    torchfort_rl_on_policy_create_system("test", temp_config_file.c_str(), TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  
    // try a prediction
    torchfort_rl_on_policy_evaluate("test", state, state_shape.size(), state_shape.data(), action, action_shape.size(), action_shape.data(), reward, 1, reward_shape.data(), TORCHFORT_FLOAT, 0);
    
    // Clean up
    delete[] state;
    delete[] action;
    std::filesystem::remove(temp_config_file);
}

INSTANTIATE_TEST_SUITE_P(
    TestShapes,   // Your instance label
    TorchFortInterface,      // Test fixture class
    ::testing::Values(
        std::vector<std::vector<int64_t>>{{1}, {1}},
        std::vector<std::vector<int64_t>>{{16}, {2}},
        std::vector<std::vector<int64_t>>{{1, 16}, {1, 2}},
        std::vector<std::vector<int64_t>>{{4, 16}, {4, 2}},
        std::vector<std::vector<int64_t>>{{1, 16}, {1}},
        std::vector<std::vector<int64_t>>{{16}, {1, 4}}
    )
);

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
  
    return RUN_ALL_TESTS();
}