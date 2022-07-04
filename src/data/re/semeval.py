from typing import Dict, TypeVar

from datasets import load_dataset

from .base import DatasetForRE

T = TypeVar("T")


class SemEvalForRE(DatasetForRE):
    HUGGINGFACE_DATASET_NAME = "sem_eval_2010_task_8"

    def __init__(self, hf_data):
        super(SemEvalForRE, self).__init__()
        self._data = hf_data

    @classmethod
    def get_dataset_name(cls):
        return cls.HUGGINGFACE_DATASET_NAME

    @property
    def inner_data(self):
        """
        Example of each row:
            {'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
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
