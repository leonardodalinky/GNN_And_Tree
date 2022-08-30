from typing import Dict, TypeVar

from ..base import DatasetBase

T = TypeVar("T")


class DatasetForEE(DatasetBase):
    def __init__(self):
        super(DatasetForEE, self).__init__()

    @classmethod
    def get_children_classes():
        raise NotImplementedError
