import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn = GCNConv(768, 768)

    def forward(self, x, edge_index):
        # x: (batch_seq_len, hidden_size)
        # edge_index: (2, batch_edges)
        x = self.gcn(x, edge_index)
        return x  # (batch_size, seq_len, hidden_size)
