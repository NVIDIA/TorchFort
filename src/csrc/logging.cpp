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
