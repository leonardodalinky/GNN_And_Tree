import torch.nn as nn


class GNNBase(nn.Module):
    GNN_NAME = "gnn_base"

    def __init__(self):
        super().__init__()
