import torch.nn as nn
from transformers import BertModel


class TreeBase(nn.Module):
    TREE_NAME = "tree_base"
    EMBEDDING_SIZE = None

    def __init__(self):
        super(TreeBase, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, *args, task_type=None):
        assert task_type in ["ner", "re", "ee"]
        if task_type == "ner":
            return self.forward_ner(*args)
        elif task_type == "re":
            return self.forward_re(*args)
        elif task_type == "ee":
            return self.forward_ee(*args)

    def forward_ner(*args):
        raise NotImplementedError

    def forward_re(*args):
        raise NotImplementedError

    def forward_ee(*args):
        raise NotImplementedError
