#include <vector>

#include <torch/torch.h>

#include "internal/models.h"
#include "internal/param_map.h"
#include "internal/setup.h"

namespace torchfort {

// SACMLP model in C++ using libtorch
void ActorCriticMLPModel::setup(const ParamMap& params) {
  // Extract params from input map.
  std::set<std::string> supported_params{"dropout",           "encoder_layer_sizes",   "actor_layer_sizes",
                                         "value_layer_sizes", "state_dependent_sigma", "log_sigma_init"};
  check_params(supported_params, params.keys());

  dropout = params.get_param<double>("dropout", 0.0)[0];
  encoder_layer_sizes = params.get_param<int>("encoder_layer_sizes");
  actor_layer_sizes = params.get_param<int>("actor_layer_sizes");
  value_layer_sizes = params.get_param<int>("value_layer_sizes");
  state_dependent_sigma = params.get_param<bool>("state_dependent_sigma", true)[0];
  double log_sigma_init = params.get_param<double>("log_sigma_init", 0.)[0];

  // sanity checks
  // make sure that value function is emitting a scalar.
  if (value_layer_sizes[value_layer_sizes.size() - 1] != 1) {
    THROW_INVALID_USAGE("ActorCriticMLPModel::setup: error, the value of the last element of value_layer_sizes has to "
                        "be equal to one.");
  }

  // Construct and register submodules.
  for (int i = 0; i < encoder_layer_sizes.size() - 1; ++i) {
    encoder_layers.push_back(register_module("encoder_fc_" + std::to_string(i),
                                             torch::nn::Linear(encoder_layer_sizes[i], encoder_layer_sizes[i + 1])));
    encoder_biases.push_back(
        register_parameter("encoder_b_" + std::to_string(i), torch::zeros(encoder_layer_sizes[i + 1])));
  }
  int encoder_last_layer_size = encoder_layer_sizes[encoder_layer_sizes.size() - 1];

  // actor
  actor_layers.push_back(
      register_module("actor_fc_entry", torch::nn::Linear(encoder_last_layer_size, actor_layer_sizes[0])));
  actor_biases.push_back(register_parameter("actor_b_entry", torch::zeros(encoder_layer_sizes[0])));
  for (int i = 0; i < actor_layer_sizes.size() - 2; ++i) {
    actor_layers.push_back(register_module("actor_fc_" + std::to_string(i),
                                           torch::nn::Linear(actor_layer_sizes[i], actor_layer_sizes[i + 1])));
    actor_biases.push_back(register_parameter("actor_b_" + std::to_string(i), torch::zeros(actor_layer_sizes[i + 1])));
  }
  int actor_last_layer_size = actor_layer_sizes[actor_layer_sizes.size() - 2];

  // mu layer
  actor_layers.push_back(register_module(
      "actor_fc_mu", torch::nn::Linear(actor_last_layer_size, actor_layer_sizes[actor_layer_sizes.size() - 1])));
  actor_biases.push_back(
      register_parameter("actor_b_mu", torch::zeros(actor_layer_sizes[actor_layer_sizes.size() - 1])));
  // sigma layer
  if (state_dependent_sigma) {
    actor_layers.push_back(
        register_module("actor_fc_log_sigma",
                        torch::nn::Linear(actor_last_layer_size, actor_layer_sizes[actor_layer_sizes.size() - 1])));
  }
  actor_biases.push_back(
      register_parameter("actor_b_log_sigma", torch::zeros(actor_layer_sizes[actor_layer_sizes.size() - 1])));

  // value
  value_layers.push_back(
      register_module("value_fc_entry", torch::nn::Linear(encoder_last_layer_size, value_layer_sizes[0])));
  value_biases.push_back(register_parameter("value_b_entry", torch::zeros(value_layer_sizes[0])));
  for (int i = 0; i < value_layer_sizes.size() - 1; ++i) {
    value_layers.push_back(register_module("value_fc_" + std::to_string(i),
                                           torch::nn::Linear(value_layer_sizes[i], value_layer_sizes[i + 1])));
    value_biases.push_back(register_parameter("value_b_" + std::to_string(i), torch::zeros(value_layer_sizes[i + 1])));
  }
}

// Implement the forward function.
std::vector<torch::Tensor> ActorCriticMLPModel::forward(const std::vector<torch::Tensor>& inputs) {
  // concatenate inputs
  auto x = torch::cat(inputs, 1);
  x = x.reshape({x.size(0), -1});

  for (int i = 0; i < encoder_layer_sizes.size() - 1; ++i) {
    // encoder part
    x = torch::relu(encoder_layers[i]->forward(x) + encoder_biases[i]);
    x = torch::dropout(x, dropout, is_training());
  }

  // actor
  torch::Tensor act = x;
  for (int i = 0; i < actor_layer_sizes.size() - 2; ++i) {
    // encoder part
    act = torch::tanh(actor_layers[i]->forward(act) + actor_biases[i]);
    act = torch::dropout(act, dropout, is_training());
  }

  torch::Tensor mu, log_sigma;
  if (state_dependent_sigma) {
    mu = actor_layers[actor_layer_sizes.size() - 2]->forward(act) + actor_biases[actor_layer_sizes.size() - 2];
    log_sigma = actor_layers[actor_layer_sizes.size() - 1]->forward(act) + actor_biases[actor_layer_sizes.size() - 1];
  } else {
    mu = actor_layers[actor_layer_sizes.size() - 1]->forward(act) + actor_biases[actor_layer_sizes.size() - 2];
    auto batch_size = mu.sizes()[0];
    log_sigma = torch::tile(actor_biases[actor_layer_sizes.size() - 1], {batch_size, 1});
  }

  // value
  torch::Tensor q = x;
  for (int i = 0; i < value_layer_sizes.size() - 1; ++i) {
    q = torch::tanh(value_layers[i]->forward(q) + value_biases[i]);
    q = torch::dropout(q, dropout, is_training());
  }
  q = value_layers[value_layer_sizes.size() - 1]->forward(q) + value_biases[value_layer_sizes.size() - 1];

  return std::vector<torch::Tensor>{mu, log_sigma, q};
}

} // namespace torchfort
