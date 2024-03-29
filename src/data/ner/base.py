from typing import Dict, TypeVar

from ..base import DatasetBase

T = TypeVar("T")


class DatasetForNER(DatasetBase):
    NER_TAG_2_IDX = dict()
    NER_IDX_2_TAG = dict()
    CLASSES_NUM = None
    IGNORED_CLASS_INDEX = None

    def __init__(self):
        super(DatasetForNER, self).__init__()

    @classmethod
    def get_children_classes(cls):
        from .conll2003 import CoNLL2003ForNER
        from .ontonotesv5 import OntoNotesv5ForNER

        return [CoNLL2003ForNER, OntoNotesv5ForNER]
