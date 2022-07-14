import torch.nn as nn
from transformers import BertModel


class TreeBase(nn.Module):
    def __init__(self):
        super(TreeBase, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained("bert-base-uncased")
