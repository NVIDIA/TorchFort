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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#include "internal/exceptions.h"
#include "internal/logging.h"

namespace torchfort {
namespace logging {

std::mutex logging_mutex;
static std::unique_ptr<std::ofstream> logfile;

std::string log_level_prefix(level log_level) {
  if (log_level == level::info) {
    return "TORCHFORT::INFO:";
  } else if (log_level == level::warn) {
    return "TORCHFORT::WARN:";
  } else if (log_level == level::error) {
    return "TORCHFORT::ERROR:";
  } else if (log_level == level::wandb) {
    return "TORCHFORT::WANDB:";
  } else {
    THROW_INVALID_USAGE("Unknown log level encountered.");
  }
}

void print(const std::string& message, level log_level) {
  std::cout << log_level_prefix(log_level) << " ";
  std::cout << message << std::endl;
}

bool open_logfile(const std::filesystem::path& filename) {

  // check if filename is empty, meaning we do not want to log
  if (filename.empty()) {
    return false;
  }

  // check if path exists
  if (filename.has_parent_path()) {
    auto path = filename.parent_path();
    std::filesystem::create_directories(path);
  }

  logfile = std::make_unique<std::ofstream>(filename, std::ofstream::out | std::ofstream::app);

  return true;
}

void write(const std::filesystem::path& filename, const std::string& message, level log_level) {
  std::lock_guard<std::mutex> guard(logging_mutex);

  // check of logfile if already open
  if (logfile == nullptr) {
    if (!open_logfile(filename))
      return;
  }
  auto line = log_level_prefix(log_level) + " " + message + "\n";
  logfile->write(line.c_str(), line.size());
  logfile->flush();
}

} // namespace logging
} // namespace torchfort
