#pragma once
#include <torch/torch.h>

// this file contains some pre-defined environments useful for debugging RL systems
class Environment {
public:
  Environment(const Environment&) = delete;

  // default constructor
  Environment() : num_steps_(0) {}

  virtual std::tuple<torch::Tensor, float> step(torch::Tensor) = 0;
  
protected:
  unsigned int num_steps_;
};

// this is the simplest of the environments, it always emits the same reward no matter what
class ConstantRewardEnvironment : public Environment, public std::enable_shared_from_this<Environment> {
public:

  ConstantRewardEnvironment(torch::IntArrayRef state_shape, torch::IntArrayRef action_shape, float default_reward) : default_reward_(default_reward),
														     state_shape_(state_shape), action_shape_(action_shape) {}
  
  std::tuple<torch::Tensor, float> step(torch::Tensor action) {
    num_steps_++;
    return std::make_tuple(torch::zeros(state_shape_), default_reward_);
  }
  
private:
  float default_reward_;
  torch::IntArrayRef state_shape_;
  torch::IntArrayRef action_shape_;
};
