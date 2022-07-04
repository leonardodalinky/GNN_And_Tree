from typing import Dict, TypeVar

from ..base import DatasetBase

T = TypeVar("T")


class DatasetForRE(DatasetBase):
    def __init__(self):
        super(DatasetForRE, self).__init__()

    @classmethod
    def load(cls: T, dataset_name: str) -> Dict[str, T]:
        from .semeval import SemEvalForRE

        clss = [SemEvalForRE]
        names = [c.get_dataset_name() for c in clss]
        for c, name in zip(clss, names):
            if name == dataset_name:
                return c.load()
        raise ValueError(f"Dataset {dataset_name} not found, available datasets: {names}")
