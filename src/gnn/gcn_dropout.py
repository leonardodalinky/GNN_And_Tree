import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from .base import GNNBase


class GCN_Dropout(GNNBase):
    GNN_NAME = "gcn_dropout"

    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(768, 768)
        self.gcn2 = GCNConv(768, 768)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        # x: (batch_seq_len, hidden_size)
        # edge_index: (2, batch_edges)
        x = self.gcn1(x, edge_index)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        return x  # (batch_seq_len, hidden_size)
