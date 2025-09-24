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
