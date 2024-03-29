from typing import Dict, Type, TypeVar

from ..base import DatasetBase

T = TypeVar("T")


class DatasetForRE(DatasetBase):
    CLASSES_NUM = None
    IGNORED_CLASS_INDEX = None

    def __init__(self):
        super(DatasetForRE, self).__init__()

    @classmethod
    def get_children_classes(cls):
        from .tacred import TacredForRE
        from .semeval import SemEvalForRE

        return [SemEvalForRE, TacredForRE]
