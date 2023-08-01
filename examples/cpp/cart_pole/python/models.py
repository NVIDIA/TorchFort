# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
import torch
from torch import nn
import torch.nn.functional as F

def weight_init(model, scale=0.02):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                sqrtk = math.sqrt(1./float(m.weight.shape[1]))
                nn.init.uniform_(m.weight, a=-sqrtk, b=sqrtk)
                if m.bias is not None:
                    m.bias.data.zero_()

class PolicyFunc(nn.Module):
    def __init__(self, hidden_features=128):
        super(PolicyFunc, self).__init__()

        layers = [nn.Linear(in_features = 4,
                            out_features = hidden_features,
                            bias=True),
                  nn.ReLU(),
                  nn.Linear(in_features = hidden_features,
                            out_features = hidden_features // 2,
                            bias=True),
                  nn.ReLU(),
                  nn.Linear(in_features = hidden_features // 2,
                            out_features = 1,
                            bias=True),
                  nn.Tanh()]

        self.fwd = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fwd(x)

class ValueFunc(nn.Module):
    def __init__(self, hidden_features=128):
        super(ValueFunc, self).__init__()

        layers = [nn.Linear(in_features = 5,
                            out_features = hidden_features,
                            bias=True),
                  nn.ReLU(),
                  nn.Linear(in_features = hidden_features,
                            out_features = hidden_features // 2,
                            bias=True),
                  nn.ReLU(),
                  nn.Linear(in_features = hidden_features // 2,
                            out_features = 1,
                            bias=True)]

        self.fwd = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)
        return self.fwd(x)
