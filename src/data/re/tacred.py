import json
from typing import Dict, TypeVar
from pathlib import Path

from datasets import load_dataset

from .base import DatasetForRE

T = TypeVar("T")


class TacredForRE(DatasetForRE):
    LOCAL_DATASET_NAME = "tacred"

    def __init__(self):
        super(SemEvalForRE, self).__init__()

    @classmethod
    def get_dataset_name(cls):
        return cls.LOCAL_DATASET_NAME

    @property
    def inner_data(self):
        """
        Example of each row:
            {
                "id": "e7798fb926b9403cfcd2", "docid": "APW_ENG_20101103.0539", "relation": "per:title",
                "token": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "."],
                "subj_start": 8, "subj_end": 9, "obj_start": 12, "obj_end": 12, "subj_type": "PERSON", "obj_type": "TITLE",
                "stanford_pos": ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG", "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."],
                "stanford_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
                "stanford_head": [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12],
                "stanford_deprel": ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj","aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct"]
            }
        """
        return self._data

    @classmethod
    def load(cls: T, dataset_name=None, **kwargs) -> Dict[str, T]:
        assert dataset_name is None or dataset_name == cls.LOCAL_DATASET_NAME
        assert kwargs["local"] == True
        assert kwargs["local_path"] is not None
        json_dir = Path(kwargs["local_path"]) / "json"
        assert json_dir.exists()
        d = {
            "train.json": "train",
            "dev.json": "validation",
            "test.json": "test",
        }
        ret = {}
        for filename, dataset_name in d.items():
            json_path = json_dir / filename
            assert json_path.exists()
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            ret[dataset_name] = data
        return ret

    @staticmethod
    def _transform_to_task_specific_format(self, item):
        # TODO
        return item
