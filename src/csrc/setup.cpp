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

#include <algorithm>
#include <any>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "internal/exceptions.h"
#include "internal/logging.h"
#include "internal/losses.h"
#include "internal/models.h"
#include "internal/param_map.h"
#include "internal/setup.h"

namespace torchfort {

void check_params(const std::set<std::string>& supported_params, const std::set<std::string>& provided_params) {
  std::set<std::string> supported_params_sanitized;
  std::transform(supported_params.begin(), supported_params.end(),
                 std::inserter(supported_params_sanitized, supported_params_sanitized.begin()),
                 [](const std::string& s) { return sanitize(s); });

  std::set<std::string> diff;
  std::set_difference(provided_params.begin(), provided_params.end(), supported_params_sanitized.begin(),
                      supported_params_sanitized.end(), std::inserter(diff, diff.begin()));

  if (diff.size() != 0) {
    std::stringstream os;
    os << "Unknown parameter(s) detected: ";
    int count = 0;
    for (const auto& entry : diff) {
      os << entry;
      if (count < diff.size() - 1) {
        os << ", ";
      }
      count++;
    }

    os << "; Supported parameter(s) are: ";
    count = 0;
    for (const auto& entry : supported_params) {
      os << entry;
      if (count < supported_params.size() - 1) {
        os << ", ";
      }
      count++;
    }
    THROW_INVALID_USAGE(os.str());
  }
}

ParamMap get_params(const YAML::Node& params_node) {
  ParamMap params;
  for (const auto& entry : params_node) {
    auto parameter_name = sanitize(entry.first.as<std::string>());
    std::vector<std::string> parameter_entry;
    try {
      parameter_entry = entry.second.as<std::vector<std::string>>();
    } catch (YAML::TypedBadConversion<std::vector<std::string>>& e) {
      // assume string, which should always work: pack it into a vector
      // for compatibility
      std::vector<std::string> values = {entry.second.as<std::string>()};
      params.add_param(parameter_name, values);
      continue;
    }
    params.add_param(parameter_name, parameter_entry);
  }
  return params;
}

std::shared_ptr<ModelWrapper> get_model(const YAML::Node& model_node) {
  auto model_type = sanitize(model_node["type"].as<std::string>());

  std::shared_ptr<ModelWrapper> model = nullptr;
  if (model_type == "torchscript") {
    auto model_params = get_params(model_node["parameters"]);
    try {
      auto torchscript_fname = model_params.get_param<std::string>("filename")[0];
      model = std::make_shared<ModelWrapper>(torchscript_fname);
    } catch (std::out_of_range) {
      THROW_INVALID_USAGE("filename parameter is required for torchscript model type.");
    }

  } else {
    std::shared_ptr<BaseModel> m = nullptr;
    try {
      m = model_registry.at(model_type)();
    } catch (std::out_of_range) {
      std::stringstream os;
      os << "Unknown model type " << model_type << " requested. Model types available are: torchscript, ";
      int count = 0;
      for (const auto& entry : model_registry) {
        os << entry.first;
        if (count < model_registry.size() - 1) {
          os << ", ";
        }
        count++;
      }
      THROW_INVALID_USAGE(os.str());
    }

    auto model_params = get_params(model_node["parameters"]);
    m->setup(model_params);
    model = std::make_shared<ModelWrapper>(m);
  }

  return model;
}

std::shared_ptr<BaseLoss> get_loss(const YAML::Node& loss_node) {
  auto loss_name = sanitize(loss_node["type"].as<std::string>());
  std::shared_ptr<BaseLoss> loss = nullptr;
  try {
    loss = loss_registry.at(loss_name)();
  } catch (std::out_of_range) {
    std::stringstream os;
    os << "Unregistered loss name " << loss_name << " requested. Registered losses available are: ";
    int count = 0;
    for (const auto& entry : loss_registry) {
      os << entry.first;
      if (count < loss_registry.size() - 1) {
        os << ", ";
      }
      count++;
    }
    THROW_INVALID_USAGE(os.str());
  }

  auto loss_params = get_params(loss_node["parameters"]);
  loss->setup(loss_params);

  return loss;
}

std::shared_ptr<torch::optim::Optimizer> get_optimizer(const YAML::Node& optimizer_node,
                                                       const std::shared_ptr<ModelWrapper>& model) {
  auto parameters = model->parameters();
  return get_optimizer(optimizer_node, parameters);
}

std::shared_ptr<torch::optim::Optimizer> get_optimizer(const YAML::Node& optimizer_node,
                                                       std::vector<torch::Tensor> parameters) {
  // const std::shared_ptr<ModelWrapper>& model) {
  auto type = sanitize(optimizer_node["type"].as<std::string>());
  auto params = get_params(optimizer_node["parameters"]);

  std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;

  if (type == "adam") {
    std::set<std::string> supported_params{"learning_rate", "beta1", "beta2", "weight_decay", "eps", "amsgrad"};
    check_params(supported_params, params.keys());
    auto options = torch::optim::AdamOptions();
    try {
      double lr = params.get_param<double>("learning_rate")[0];
      options = options.lr(lr);
    } catch (std::out_of_range) {
      // use default
    }

    try {
      double beta1 = params.get_param<double>("beta1")[0];
      double beta2 = params.get_param<double>("beta2")[0];
      options.betas(std::make_tuple(beta1, beta2));
    } catch (std::out_of_range) {
      // use default
    }

    try {
      double weight_decay = params.get_param<double>("weight_decay")[0];
      options.weight_decay(weight_decay);
    } catch (std::out_of_range) {
      // use default
    }

    try {
      double eps = params.get_param<double>("eps")[0];
      options.eps(eps);
    } catch (std::out_of_range) {
      // use default
    }

    try {
      bool amsgrad = params.get_param<bool>("amsgrad")[0];
      options.amsgrad(amsgrad);
    } catch (std::out_of_range) {
      // use default
    }

    optimizer = std::shared_ptr<torch::optim::Optimizer>(new torch::optim::Adam(parameters, options));
  } else if (type == "sgd") {
    std::set<std::string> supported_params{"learning_rate", "momentum", "dampening", "weight_decay", "nesterov"};
    check_params(supported_params, params.keys());
    double lr;
    try {
      lr = params.get_param<double>("learning_rate")[0];
    } catch (std::out_of_range) {
      lr = 0.001; // default
    }

    auto options = torch::optim::SGDOptions(lr);

    try {
      double momentum = params.get_param<double>("momentum")[0];
      options = options.momentum(momentum);
    } catch (std::out_of_range) {
      // use default
    }

    try {
      double dampening = params.get_param<double>("dampening")[0];
      options = options.dampening(dampening);
    } catch (std::out_of_range) {
      // use default
    }

    try {
      double weight_decay = params.get_param<double>("weight_decay")[0];
      options = options.weight_decay(weight_decay);
    } catch (std::out_of_range) {
      // use default
    }

    try {
      bool nesterov = params.get_param<bool>("nesterov")[0];
      options = options.nesterov(nesterov);
    } catch (std::out_of_range) {
      // use default
    }

    optimizer = std::shared_ptr<torch::optim::Optimizer>(new torch::optim::SGD(parameters, options));

  } else {
    THROW_INVALID_USAGE("Unknown optimizer type provided.");
  }

  return optimizer;
}

std::shared_ptr<ModelState> get_state(const char* name, const YAML::Node& state_node) {
  auto state = std::make_shared<ModelState>();

  if (state_node["general"]) {
    auto params = get_params(state_node["general"]);
    std::set<std::string> supported_params{"report_frequency", "enable_wandb_hook", "verbose"};
    check_params(supported_params, params.keys());
    state->report_frequency = params.get_param<int>("report_frequency")[0];
    try {
      state->enable_wandb_hook = params.get_param<bool>("enable_wandb_hook")[0];

      if (state->enable_wandb_hook) {
        // here is a good place to query the env variable for the logfile directory:
        char* env_p = std::getenv("TORCHFORT_LOGDIR");
        std::filesystem::path logdir;
        if (env_p) {
          logdir = static_cast<std::filesystem::path>(std::string(env_p));
        }
        // we should copy the model yaml file into the logging directory
        if (!logdir.empty()) {
          std::filesystem::path logbase = "torchfort.log";
          state->report_file = (logdir / logbase);

          auto config = state_node;
          config["identifier"] = name;
          auto configfilename =
              logdir / static_cast<std::filesystem::path>("config_" + filename_sanitize(name) + ".yaml");
          std::ofstream fout(configfilename);
          fout << config;
          fout.close();
        } else {
          std::stringstream os;
          os << "enable_wandb_hook is true but the environment variable TORCHFORT_LOGDIR was not specified. "
                "To enable logging, set TORCHFORT_LOGDIR to a writeable directory prior to launching wandb_helper.py "
                "and your torchfort enabled application.";
          torchfort::logging::print(os.str(), torchfort::logging::warn);
        }
      }
    } catch (std::out_of_range) {
      // do nothing
      state->enable_wandb_hook = false;
    }

    try {
      state->verbose = params.get_param<bool>("verbose")[0];
    } catch (std::out_of_range) {
      state->verbose = false;
    }
  }

  return state;
}

} // namespace torchfort
