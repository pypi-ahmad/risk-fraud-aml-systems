"""
models.py — Graph neural networks for node-level AML risk scoring.

Two transductive, full-batch architectures on the Elliptic graph:

  * GraphSAGE — mean-aggregator inductive convolutions; a strong, stable default.
  * GAT       — attention-weighted aggregation; lets the model weigh suspicious
                neighbours more heavily (interpretable attention as a bonus).

Both output a single logit per node (illicit vs licit). We expose a `embed()` hook
that returns the penultimate representation so the anomaly branch and the hybrid
meta-model can reuse the learned embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
        self.head = nn.Linear(hidden, 1)

    def embed(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        h = self.embed(x, edge_index)
        return self.head(h).squeeze(-1)      # [N] logits


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_d = in_dim
        for i in range(num_layers):
            last = i == num_layers - 1
            out_heads = 1 if last else heads
            conv = GATConv(in_d, hidden, heads=out_heads,
                           concat=not last, dropout=dropout)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden * (out_heads if not last else 1)))
            in_d = hidden * (out_heads if not last else 1)
        self.head = nn.Linear(in_d, 1)

    def embed(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        h = self.embed(x, edge_index)
        return self.head(h).squeeze(-1)


def build_model(name: str, in_dim: int, **kw) -> nn.Module:
    name = name.lower()
    if name in ("sage", "graphsage"):
        return GraphSAGE(in_dim, **kw)
    if name == "gat":
        return GAT(in_dim, **kw)
    raise ValueError(f"unknown model '{name}' (use 'sage' or 'gat')")
