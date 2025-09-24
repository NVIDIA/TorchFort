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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif
#include <torch/script.h>
#include <torch/torch.h>

#include "internal/base_model.h"
#include "internal/defines.h"
#include "internal/model_wrapper.h"
#include "internal/utils.h"

namespace torchfort {

ModelWrapper::ModelWrapper(const std::shared_ptr<BaseModel>& model) : model{model} {}

ModelWrapper::ModelWrapper(const std::shared_ptr<torch::jit::Module>& model_jit) : model_jit{model_jit}, jit{true} {}

ModelWrapper::ModelWrapper(const std::string& jit_model_fname) : jit{true} {

  if (!std::filesystem::exists(jit_model_fname)) {
    THROW_INVALID_USAGE(jit_model_fname + " does not exist.");
  }

  model_jit = std::shared_ptr<torch::jit::Module>(new torch::jit::Module);
  *model_jit = torch::jit::load(jit_model_fname, device_);
}

std::vector<torch::Tensor> ModelWrapper::parameters() const {
  if (jit) {
    std::vector<torch::Tensor> parameters;
    for (const auto& params : model_jit->parameters()) {
      parameters.push_back(params);
    }
    return parameters;
  }

  return model->parameters();
}

torch::OrderedDict<std::string, torch::Tensor> ModelWrapper::named_parameters() const {
  if (jit) {
    torch::OrderedDict<std::string, torch::Tensor> parameters;
    for (const auto& params : model_jit->named_parameters()) {
      parameters.insert(params.name, params.value);
    }
    return parameters;
  }

  return model->named_parameters();
}

void ModelWrapper::to(torch::Device device, bool non_blocking) {
  if (jit) {
    model_jit->to(device, non_blocking);
  } else {
    model->to(device, non_blocking);
  }

  this->device_ = device;
}

void ModelWrapper::train() {
  if (jit) {
    model_jit->train();
  } else {
    model->train();
  }
}

void ModelWrapper::eval() {
  if (jit) {
    model_jit->eval();
  } else {
    model->eval();
  }
}

std::vector<torch::Tensor> ModelWrapper::forward(const std::vector<torch::Tensor>& inputs) const {
  if (jit) {
    std::vector<torch::jit::IValue> inputs_jit;
    inputs_jit.assign(inputs.begin(), inputs.end());
    auto result = model_jit->forward(inputs_jit);
    if (result.isTensor()) {
      return std::vector<torch::Tensor>{result.toTensor()};
    } else if (result.isTuple()) {
      std::vector<torch::Tensor> tensors;
      for (const auto& x : result.toTuple()->elements()) {
        tensors.push_back(x.toTensor());
      }
      return tensors;
    } else {
      assert(true);
    }
  }
  return model->forward(inputs);
}

void ModelWrapper::save(const std::string& fname) const {
  if (jit) {
    model_jit->save(fname);
  } else {
    torch::save(model, fname);
  }
}

void ModelWrapper::load(const std::string& fname) {
  if (!std::filesystem::exists(fname)) {
    THROW_INVALID_USAGE(fname + " does not exist.");
  }
  if (jit) {
    model_jit.reset();
    model_jit = std::shared_ptr<torch::jit::Module>(new torch::jit::Module);

    *model_jit = torch::jit::load(fname, device_);
  } else {
    torch::load(model, fname, device_);
  }
}

torch::Device ModelWrapper::device() const { return device_; }

} // namespace torchfort
