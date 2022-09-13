import torch
import torch.nn as nn

from .ner_normal import NerNormal


class NerNormal_v2(NerNormal):
    FRAMEWORK_NAME = "ner_normal_v2"

    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(self.tree.EMBEDDING_SIZE, self.tree.EMBEDDING_SIZE // 4),
            nn.GELU(),
            nn.Linear(self.tree.EMBEDDING_SIZE // 4, self.classes_num),
        )
