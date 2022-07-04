from typing import Dict, TypeVar

from ..base import DatasetBase

T = TypeVar("T")


class DatasetForNER(DatasetBase):
    def __init__(self):
        super(DatasetForNER, self).__init__()

    @classmethod
    def load(cls: T, dataset_name: str) -> Dict[str, T]:
        from .conll2003 import CoNLL2003ForNER
        from .ontonotesv5 import OntoNotesv5ForNER

        clss = [CoNLL2003ForNER, OntoNotesv5ForNER]
        names = [c.get_dataset_name() for c in clss]
        for c, name in zip(clss, names):
            if name == dataset_name:
                return c.load()
        raise ValueError(f"Dataset {dataset_name} not found, available datasets: {names}")
