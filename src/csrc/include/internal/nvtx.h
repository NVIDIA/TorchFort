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

#include <string>

#ifdef ENABLE_GPU
#include <nvtx3/nvToolsExt.h>
#endif

namespace torchfort {

// Helper class for NVTX ranges
class nvtx {
public:
#ifdef ENABLE_GPU
  static void rangePush(const std::string& range_name) {
    static constexpr int ncolors_ = 8;
    static constexpr int colors_[ncolors_] = {0x3366CC, 0xDC3912, 0xFF9900, 0x109618,
                                              0x990099, 0x3B3EAC, 0x0099C6, 0xDD4477};
    std::hash<std::string> hash_fn;
    int color = colors_[hash_fn(range_name) % ncolors_];
    nvtxEventAttributes_t ev = {0};
    ev.version = NVTX_VERSION;
    ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    ev.colorType = NVTX_COLOR_ARGB;
    ev.color = color;
    ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
    ev.message.ascii = range_name.c_str();
    nvtxRangePushEx(&ev);
  }

  static void rangePop() { nvtxRangePop(); }
#else
  static void rangePush(const std::string& range_name) {}
  static void rangePop() {}
#endif
};

} // namespace torchfort
