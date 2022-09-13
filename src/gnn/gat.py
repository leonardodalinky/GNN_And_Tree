import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .base import GNNBase


class GAT(GNNBase):
    GNN_NAME = "gat"

    def __init__(self):
        super().__init__()
        self.gat1 = GATConv(768, 768)
        self.gat2 = GATConv(768, 768)

    def forward(self, x, edge_index):
        # x: (batch_seq_len, hidden_size)
        # edge_index: (2, batch_edges)
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return x  # (batch_seq_len, hidden_size)
