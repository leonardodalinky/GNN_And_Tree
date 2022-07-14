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
