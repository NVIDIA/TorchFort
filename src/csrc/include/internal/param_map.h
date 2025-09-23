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

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "internal/exceptions.h"
#include "internal/utils.h"

namespace torchfort {

// Helper function to get type as string
template <typename T> std::string type_string() {
  if (std::is_same<T, int>::value) {
    return "int";
  } else if (std::is_same<T, float>::value) {
    return "float";
  } else if (std::is_same<T, double>::value) {
    return "double";
  } else if (std::is_same<T, bool>::value) {
    return "bool";
  }
  return "UNKNOWN";
};

// Conversion functor
template <typename T> struct ParamMapConverter {
  T operator()(const std::string& s) {
    try {
      if constexpr (std::is_same<T, int>::value) {
        return std::stoi(sanitize(s));
      }
      if constexpr (std::is_same<T, float>::value) {
        return std::stof(sanitize(s));
      }
      if constexpr (std::is_same<T, double>::value) {
        return std::stod(sanitize(s));
      }
      if constexpr (std::is_same<T, bool>::value) {
        std::string s_ = sanitize(s);
        bool val;
        if (s_ == "true") {
          val = true;
        } else if (s_ == "false") {
          val = false;
        } else {
          val = std::stoi(s_);
        }
        return val;
      }
      if constexpr (std::is_same<T, std::string>::value) {
        return s;
      }
    } catch (std::invalid_argument) {
      THROW_INVALID_USAGE("Could not convert provided parameter value " + s + " to required type.");
    }

    THROW_INTERNAL_ERROR("Unknown conversion type.");
  }
};

class ParamMap {
public:
  template <typename T> void add_param(const std::string& key, const std::vector<T>& value);

  template <typename T> std::vector<T> get_param(const std::string& key) const;

  template <typename T> std::vector<T> get_param(const std::string& key, const T& defval) const;

  std::set<std::string> keys() const;

private:
  std::unordered_map<std::string, std::vector<std::string>> params;
};

template <typename T> void ParamMap::add_param(const std::string& key, const std::vector<T>& value) {
  params[sanitize(key)] = value;
}

template <typename T> std::vector<T> ParamMap::get_param(const std::string& key) const {
  const auto& entry = params.at(sanitize(key));
  std::vector<T> values;
  std::transform(entry.begin(), entry.end(), std::back_inserter(values), ParamMapConverter<T>());
  return values;
}

// parameter with default value
template <typename T> std::vector<T> ParamMap::get_param(const std::string& key, const T& defval) const {
  try {
    const auto& entry = params.at(sanitize(key));
    std::vector<T> values;
    std::transform(entry.begin(), entry.end(), std::back_inserter(values), ParamMapConverter<T>());
    return values;
  } catch (std::out_of_range) {
    return {defval};
  }
}

} // namespace torchfort
