from typing import Dict, TypeVar

from ..base import DatasetBase

T = TypeVar("T")


class DatasetForEE(DatasetBase):
    def __init__(self):
        super(DatasetForEE, self).__init__()

    @classmethod
    def load(cls: T, dataset_name: str) -> Dict[str, T]:
        # TODO: subclass this class and implement this method
        pass
