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

class CustomLoss(torch.nn.Module):
  def __init__(self):
    super(CustomLoss, self).__init__()

  def forward(self, prediction, label, node_types):

    # Compute MSE over all nodes
    err = (label - prediction)**2

    # Zero out error for boundary nodes
    mask = node_types != 0
    err *= mask.unsqueeze(-1)

    # Compute mean over non-boundary nodes
    mse = torch.sum(err) / (torch.sum(mask) * err.shape[1])

    return mse

def main():
  # Create loss module
  loss = CustomLoss()
  print("loss module:", loss)

  try:
    # Move model to GPU, JIT, and save
    loss.to("cuda")
  except:
    print("PyTorch does not have CUDA support. Saving model on CPU.")
  loss_jit = torch.jit.script(loss)
  loss_jit.save("loss_torchscript.pt")

if __name__ == "__main__":
  main()
