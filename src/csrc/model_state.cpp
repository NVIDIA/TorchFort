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

#include <string>

#include <torch/torch.h>

#include "internal/exceptions.h"
#include "internal/model_state.h"

namespace torchfort {

void ModelState::save(const std::string& fname) {
  torch::serialize::OutputArchive archive;
  archive.write("step_train", torch::IValue(step_train));
  archive.write("step_inference", torch::IValue(step_inference));
  archive.write("device", torch::IValue(device));
  archive.save_to(fname);
}

void ModelState::load(const std::string& fname) {
  if (!std::filesystem::exists(fname)) {
    THROW_INVALID_USAGE(fname + " does not exist.");
  }

  torch::serialize::InputArchive archive;
  archive.load_from(fname);

  torch::IValue ivalue;
  if (!archive.try_read("step_train", ivalue)) {
    THROW_INVALID_USAGE(fname + " is missing required data.");
  }
  step_train = ivalue.to<int64_t>();

  if (!archive.try_read("step_inference", ivalue)) {
    THROW_INVALID_USAGE(fname + " is missing required data.");
  }
  step_inference = ivalue.to<int64_t>();

  if (!archive.try_read("device", ivalue)) {
    THROW_INVALID_USAGE(fname + " is missing required data.");
  }
  device = ivalue.to<torch::Device>();
}

} // namespace torchfort
