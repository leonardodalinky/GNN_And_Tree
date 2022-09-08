from typing import Dict

import torch
from torch.utils.data import Dataset


class DictTensorDataset(Dataset):
    def __init__(self, d: Dict[str, torch.Tensor]):
        assert len(d) != 0
        self.b = None
        for name, tensor in d.items():
            self.b = self.b or int(tensor.shape[0])
            assert (
                tensor.shape[0] == self.b
            ), f"Batch shape mismatch: get tensor shape {tensor.shape} but prefer batch size {b}."
        self.dict_data = d

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.dict_data.items()}

    def __len__(self):
        return self.b
