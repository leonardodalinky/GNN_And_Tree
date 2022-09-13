import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

from .base import GNNBase


class GIN(GNNBase):
    GNN_NAME = "gin"

    def __init__(self):
        super().__init__()
        self.gin1 = GINConv(self._make_mlp())
        self.gin2 = GINConv(self._make_mlp())

    def forward(self, x, edge_index):
        # x: (batch_seq_len, hidden_size)
        # edge_index: (2, batch_edges)
        x = self.gin1(x, edge_index)
        x = self.gin2(x, edge_index)
        return x  # (batch_seq_len, hidden_size)

    @classmethod
    def _make_mlp(cls) -> nn.Module:
        return nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
            nn.SiLU(),
        )
