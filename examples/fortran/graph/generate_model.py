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

import torch

class MessagePassing(torch.nn.Module):
  def __init__(self, hidden_dim):
    super(MessagePassing, self).__init__()

    self.mlp_edge = torch.nn.Sequential(torch.nn.Linear(3*hidden_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim),
                                        torch.nn.LayerNorm(hidden_dim))
    self.mlp_node = torch.nn.Sequential(torch.nn.Linear(2*hidden_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim),
                                        torch.nn.LayerNorm(hidden_dim))

  def forward(self, edge_idx, node_feats, edge_feats):
    senders = edge_idx[:,0]
    receivers = edge_idx[:,1]

    edge_update = torch.cat([node_feats[senders], node_feats[receivers], edge_feats], dim=-1)
    edge_update = self.mlp_edge(edge_update)

    accumulate_edges = torch.zeros([node_feats.shape[0], edge_feats.shape[1]], dtype=edge_feats.dtype, device=edge_feats.device)
    receivers = receivers.unsqueeze(-1).expand(-1, edge_feats.shape[1])
    accumulate_edges = torch.scatter_add(accumulate_edges, src=edge_feats, index=receivers, dim=0)
    node_update = torch.cat([node_feats, accumulate_edges], dim=-1)
    node_update = self.mlp_node(node_update)

    edge_feats = edge_feats + edge_update
    node_feats = node_feats + node_update

    return node_feats, edge_feats


class Net(torch.nn.Module):
  def __init__(self, in_node_features, in_edge_features, hidden_dim, n_message_passing_steps):
    super(Net, self).__init__()
    self.encoder_node = torch.nn.Sequential(torch.nn.Linear(in_node_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim),
                                            torch.nn.LayerNorm(hidden_dim))
    self.encoder_edge = torch.nn.Sequential(torch.nn.Linear(in_edge_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim),
                                            torch.nn.LayerNorm(hidden_dim))

    self.mp_layers = torch.nn.ModuleList()
    for _ in range(n_message_passing_steps):
      self.mp_layers.append(MessagePassing(hidden_dim))

    self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_dim, in_node_features))

  def forward(self, edge_idx, node_feats, edge_feats):
    # Encode node and edge features
    node_feats = self.encoder_node(node_feats)
    edge_feats = self.encoder_edge(edge_feats)

    # Message passing
    for mp in self.mp_layers:
      node_feats, edge_feats = mp(edge_idx, node_feats, edge_feats)

    # Decode node featues
    node_feats = self.decoder(node_feats)

    return node_feats


def main():
  # Create model
  model = Net(1, 3, 128, 8)
  print("graph model:", model)

  try:
    # Move model to GPU, JIT, and save
    model.to("cuda")
  except:
    print("PyTorch does not have CUDA support. Saving model on CPU.")
  model_jit = torch.jit.script(model)
  model_jit.save("model_torchscript.pt")

if __name__ == "__main__":
  main()
