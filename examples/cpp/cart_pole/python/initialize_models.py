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


import argparse as ap
import math
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F

from models import weight_init, PolicyFunc, ValueFunc

def main(args):

    # set seed
    torch.manual_seed(666)

    # CUDA check
    if torch.cuda.is_available():
        torch.cuda.manual_seed(666)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # parameters
    batch_size = 64

    # policy model
    pmodel = PolicyFunc(hidden_features=args.num_hidden_features).to(device)
    weight_init(pmodel)
    jpmodel = torch.jit.script(pmodel)
    inp = torch.ones((batch_size, 4), dtype=torch.float32, device=device)
    out = jpmodel(inp)
    print("Policy model:", pmodel)
    print("Policy model output shape:", out.shape)
    torch.jit.save(jpmodel, "policy.pt")

    # value model
    qmodel = ValueFunc(hidden_features=args.num_hidden_features).to(device)
    weight_init(qmodel)
    jqmodel = torch.jit.script(qmodel)
    inp_a = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
    out = jqmodel(inp, inp_a)
    print("Value model:", qmodel)
    print("Value model output shape:", out.shape)
    torch.jit.save(jqmodel, "value.pt")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--num_hidden_features", type=int, default=128, help="Number of hidden features")
    args = parser.parse_args()

    main(args)
