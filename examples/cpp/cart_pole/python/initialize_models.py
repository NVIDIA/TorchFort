# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
