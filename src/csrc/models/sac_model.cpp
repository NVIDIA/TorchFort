#include <vector>

#include <torch/torch.h>

#include "internal/models.h"
#include "internal/param_map.h"
#include "internal/setup.h"

namespace torchfort {

// SACMLP model in C++ using libtorch
void SACMLPModel::setup(const ParamMap& params) {
  // Extract params from input map.
  std::set<std::string> supported_params{"dropout", "layer_sizes"};
  check_params(supported_params, params.keys());

  dropout = params.get_param<double>("dropout", 0.0)[0];
  layer_sizes = params.get_param<int>("layer_sizes");

  // Construct and register submodules.
  for (int i = 0; i < layer_sizes.size() - 1; ++i) {
    if (i < layer_sizes.size() - 2) {
      encoder_layers.push_back(register_module("encoder_fc_" + std::to_string(i), torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1])));
      biases.push_back(register_parameter("encoder_b_" + std::to_string(i), torch::zeros(layer_sizes[i + 1])));
    } else{
      // first output
      out_layers.push_back(register_module("out_fc_1_" + std::to_string(i), torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1])));
      out_biases.push_back(register_parameter("out_b_1_" + std::to_string(i), torch::zeros(layer_sizes[i + 1])));
      // second output
      out_layers.push_back(register_module("out_fc_2_" + std::to_string(i), torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1])));
      out_biases.push_back(register_parameter("out_b_2_" + std::to_string(i), torch::zeros(layer_sizes[i + 1])));
    }
  }
}

// Implement the forward function.
std::vector<torch::Tensor> SACMLPModel::forward(const std::vector<torch::Tensor>& inputs) {
  // concatenate inputs
  auto x = torch::cat(inputs, 1);
  x = x.reshape({x.size(0), -1});
  torch::Tensor y, z;

  for (int i = 0; i < layer_sizes.size() - 1; ++i) {
    if (i < layer_sizes.size() - 2) {
      // encoder part
      x = torch::relu(encoder_layers[i]->forward(x) + biases[i]);
      x = torch::dropout(x, dropout, is_training());
    } else {
      // y
      y = out_layers[0]->forward(x) + out_biases[0];
      // z
      z = out_layers[1]->forward(x) + out_biases[1];
    }
  }
  return std::vector<torch::Tensor>{y, z};
}

} // namespace torchfort
