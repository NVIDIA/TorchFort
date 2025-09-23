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

#pragma once

#include <torch/torch.h>

#include "internal/base_model.h"

namespace torchfort {

class ModelWrapper {
public:
  ModelWrapper(const std::shared_ptr<BaseModel>& model);

  ModelWrapper(const std::shared_ptr<torch::jit::Module>& model_jit);

  ModelWrapper(const std::string& jit_model_fname);

  std::vector<torch::Tensor> parameters() const;

  torch::OrderedDict<std::string, torch::Tensor> named_parameters() const;

  void to(torch::Device device, bool non_blocking = false);

  void train();

  void eval();

  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) const;

  void save(const std::string& fname) const;

  void load(const std::string& fname);

  torch::Device device() const;

private:
  bool jit = false;
  std::shared_ptr<BaseModel> model;
  std::shared_ptr<torch::jit::Module> model_jit;
  torch::Device device_ = torch::Device(torch::kCPU);
};

} // namespace torchfort
