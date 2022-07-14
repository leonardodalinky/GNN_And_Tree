from abc import ABC, abstractmethod
from typing import Dict, TypeVar

from torch.utils.data import Dataset

T = TypeVar("T")


class DatasetBase(Dataset, ABC):
    def __init__(self):
        super(DatasetBase, self).__init__()

    def __getitem__(self, index):
        return self.inner_data[index]

    def __len__(self):
        return len(self.inner_data)

    def __str__(self):
        return str(self.inner_data)

    def __repr__(self):
        return str(self)

    @classmethod
    def load(cls: T, dataset_name: str) -> Dict[str, T]:
        pass

    @property
    @abstractmethod
    def inner_data(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_dataset_name(cls):
        raise NotImplementedError
