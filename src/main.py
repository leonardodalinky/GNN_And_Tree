import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler

from .data.re import DatasetForRE
from .model.gcn import GCN
from .tree.left_tree import LeftTree


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tree = LeftTree()
        self.gcn = GCN()

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds, edges = self.tree(x)
        return self.gcn(embeds, edges)


def train():
    pass


def main():
    torch.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    datasets = DatasetForRE.load("sem_eval_2010_task_8")
    train_dataset, val_dataset = datasets["train"], datasets["validate"]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


if __name__ == "__main__":
    main()
