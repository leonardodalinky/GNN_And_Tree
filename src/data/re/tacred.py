import os
import json
from typing import Dict, Union, TypeVar
from pathlib import Path

import torch
from transformers import BertTokenizer

from utils.data import DictTensorDataset

from .base import DatasetForRE

T = TypeVar("T")


class TacredForRE(DatasetForRE):
    LOCAL_DATASET_NAME = "tacred"
    BERT_TOKENIZER_NAME = "bert-base-uncased"
    MAX_SEQ_LEN = 160
    CLASSES_NUM = 42
    IGNORED_CLASS_INDEX = 0

    def __init__(self, data):
        super(TacredForRE, self).__init__()
        self._data = data

    @classmethod
    def get_dataset_name(cls):
        return cls.LOCAL_DATASET_NAME

    @property
    def inner_data(self):
        """
        Deprecated.

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
        dc = kwargs.get("dataset_config")
        assert dc is not None
        assert dc.get("local") == True, "Tacred dataset can only be loaded from local path."
        assert dc.get("local_path") is not None
        local_path = Path(dc["local_path"])
        # json_dir = Path(kwargs["local_path"]) / "json"
        # assert json_dir.exists()
        # d = {
        #     "train.json": "train",
        #     "dev.json": "validatate",
        #     "test.json": "test",
        # }
        # ret = {}
        # for filename, dataset_name in d.items():
        #     json_path = json_dir / filename
        #     assert json_path.exists()
        #     with open(json_path, encoding="utf-8") as f:
        #         data = json.load(f)
        #     ret[dataset_name] = data
        # return ret
        d = {
            "train": ("train_sentence.json", "train_label_id.json"),
            "validate": ("dev_sentence.json", "dev_label_id.json"),
            "test": ("test_sentence.json", "test_label_id.json"),
        }
        ret = {}
        for dataset_name, paths in d.items():
            sent_filename, label_filename = paths
            ret[dataset_name] = cls(
                cls._create_dataset(
                    local_path / sent_filename,
                    local_path / label_filename,
                )
            )
        return ret

    @classmethod
    def _create_dataset(
        cls,
        data_json_path: Union[str, bytes, os.PathLike],
        label_json_path: Union[str, bytes, os.PathLike],
    ):
        with Path(data_json_path).open("r", encoding="utf8") as f:
            sentences = json.load(f)
        with Path(label_json_path).open("r", encoding="utf8") as f:
            sentence_labels = json.load(f)
        assert len(sentences) == len(sentence_labels)

        input_ids = []
        attention_masks = []
        labels = []
        e1_pos = []
        e2_pos = []
        actual_lens = []

        # Load tokenizer.
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(cls.BERT_TOKENIZER_NAME, do_lower_case=True)

        # pre-processing sentenses to BERT pattern
        for i in range(len(sentences)):
            encoded_dict = tokenizer(
                sentences[i],  # Sentence to encode.
                add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                max_length=cls.MAX_SEQ_LEN,
                padding="max_length",
                truncation=False,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )
            try:
                ids = encoded_dict["input_ids"]
                mask = encoded_dict["attention_mask"]
                # Find e1(id:2487) and e2(id:2475) position
                pos1 = (ids == 2487).nonzero()[0][1].item()
                pos2 = (ids == 2475).nonzero()[0][1].item()
                if pos1 >= cls.MAX_SEQ_LEN:
                    pos1 = -1
                if pos2 >= cls.MAX_SEQ_LEN:
                    pos2 = -1
                # truncate manually
                if ids.shape[1] > cls.MAX_SEQ_LEN:
                    ids = torch.narrow_copy(ids, 1, 0, cls.MAX_SEQ_LEN)
                    ids[0, -1] = tokenizer.sep_token_id
                    mask = torch.narrow_copy(mask, 1, 0, cls.MAX_SEQ_LEN)
                e1_pos.append(pos1)
                e2_pos.append(pos2)
                # Add the encoded sentence to the list.
                input_ids.append(ids)
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(mask)
                labels.append(sentence_labels[i])
                actual_len = torch.max(torch.arange(1, cls.MAX_SEQ_LEN + 1, dtype=torch.long) * mask).item()
                actual_lens.append(actual_len)
            except:
                pass
                # print(sent)

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        e1_pos = torch.tensor(e1_pos)
        e2_pos = torch.tensor(e2_pos)
        actual_lens = torch.tensor(actual_lens, dtype=torch.long)

        # Combine the training inputs into a TensorDataset.
        return DictTensorDataset(
            {
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "labels": labels,
                "e1_pos": e1_pos,
                "e2_pos": e2_pos,
                "actual_lens": actual_lens,
            }
        )
