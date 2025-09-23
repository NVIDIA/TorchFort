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

import torch

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode="circular")

  def forward(self, x):
    return self.conv1(x)


def main():
  # Create model
  model = Net()
  print("FCN model:", model)

  try:
    # Move model to GPU, JIT, and save
    model.to("cuda")
  except:
    print("PyTorch does not have CUDA support. Saving model on CPU.")
  model_jit = torch.jit.script(model)
  model_jit.save("fcn_torchscript.pt")

if __name__ == "__main__":
  main()
