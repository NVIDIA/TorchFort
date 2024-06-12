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
