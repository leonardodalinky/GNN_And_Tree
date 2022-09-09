import torch
import torch.nn as nn

from .re_normal import ReNormal


class ReNormal_v2(ReNormal):
    FRAMEWORK_NAME = "re_normal_v2"

    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(self.tree.EMBEDDING_SIZE, self.tree.EMBEDDING_SIZE // 4),
            nn.GELU(),
            nn.Linear(self.tree.EMBEDDING_SIZE // 4, self.classes_num),
        )
