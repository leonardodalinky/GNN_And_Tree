import torch
from torch_geometric.data import Data, Batch

from .base import TreeBase


class LeftTree(TreeBase):
    def __init__(self):
        super(LeftTree, self).__init__()

    def forward(self, *args, task_type=None):
        assert task_type in ["ner", "re", "ee"]
        if task_type == "ner":
            return self.forward_ner(*args)
        elif task_type == "re":
            return self.forward_re(*args)
        elif task_type == "ee":
            return self.forward_ee(*args)

    def forward_re(self, *args):
        """
        Inputs:
            `args` should be TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos)
        Outputs:
            Left tree
        """
        input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens = args
        output_dict = self.bert(
            input_ids=input_ids,
            attention_mask=attention_masks,
            return_dict=True,
        )
        edges = list()
        embeds = output_dict["last_hidden_state"]  # (batch_size, seq_len, hidden_size)
        # build left tree edges
        B, _, _ = embeds.shape
        for b in range(B):
            actual_len = actual_lens[b].item()
            edge = torch.zeros((2, actual_len - 1), dtype=torch.long, device=embeds.device)
            edge[0] = torch.arange(1, actual_len, dtype=torch.long, device=embeds.device)
            edge[1] = torch.arange(0, actual_len - 1, dtype=torch.long, device=embeds.device)
            edges.append(edge)
        # construct batch
        data_list = list()
        for embed, edge in zip(embeds, edges):
            # embed: (seq_len, hidden_size), edge: (2, each_edge_size)
            data_list.append(Data(x=embed, edge_index=edge))
        return (
            embeds,
            edges,
            Batch.from_data_list(data_list),
        )  # (batch_size, seq_len, hidden_size), list(2, each_edge_size), Batch
