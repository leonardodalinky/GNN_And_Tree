from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar

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
    def load(cls: T, dataset_name: str, **kwargs) -> Dict[str, T]:
        clss = cls.get_children_classes()
        names = [c.get_dataset_name() for c in clss]
        for c, name in zip(clss, names):
            if name == dataset_name:
                return c.load(**kwargs)
        raise ValueError(f"Dataset {dataset_name} not found, available datasets: {names}")

    @classmethod
    def load_cls(cls: T, dataset_name: str) -> Type[T]:
        """
        Return class of the dataset.
        """
        clss = cls.get_children_classes()
        names = [c.get_dataset_name() for c in clss]
        for c, name in zip(clss, names):
            if name == dataset_name:
                return c
        raise ValueError(f"Dataset {dataset_name} not found, available datasets: {names}")

    @property
    @abstractmethod
    def inner_data(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_dataset_name(cls):
        raise NotImplementedError

    @classmethod
    def get_children_classes(cls):
        raise NotImplementedError
