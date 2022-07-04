from typing import Dict, TypeVar

from datasets import load_dataset

from .base import DatasetForNER

T = TypeVar("T")


class CoNLL2003ForNER(DatasetForNER):
    HUGGINGFACE_DATASET_NAME = "conll2003"

    def __init__(self, hf_data):
        super(CoNLL2003ForNER, self).__init__()
        self._data = hf_data

    @classmethod
    def get_dataset_name(cls):
        return cls.HUGGINGFACE_DATASET_NAME

    @property
    def inner_data(self):
        """
        Example of each row:
            {'id': '0', 'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], 'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7], 'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0], 'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}
        """
        return self._data

    @classmethod
    def load(cls: T, dataset_name=None) -> Dict[str, T]:
        assert dataset_name is None or dataset_name == cls.HUGGINGFACE_DATASET_NAME
        ds_dict = load_dataset(cls.HUGGINGFACE_DATASET_NAME)
        return {k: cls(v) for k, v in ds_dict.items()}

    @staticmethod
    def _transform_to_task_specific_format(item):
        return item
