from typing import Dict, TypeVar

import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from .base import DatasetForRE

T = TypeVar("T")


class SemEvalForRE(DatasetForRE):
    HUGGINGFACE_DATASET_NAME = "sem_eval_2010_task_8"
    BERT_TOKENIZER_NAME = "bert-base-uncased"
    MAX_SEQ_LEN = 128
    CLASSES_NUM = 19
    IGNORED_CLASS_INDEX = 0

    def __init__(self, hf_data):
        super(SemEvalForRE, self).__init__()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.BERT_TOKENIZER_NAME, do_lower_case=True)
        self._data = self._transform_dataset(hf_data)

    @classmethod
    def get_dataset_name(cls):
        return cls.HUGGINGFACE_DATASET_NAME

    @property
    def inner_data(self):
        return self._data

    @classmethod
    def load(cls: T, dataset_name=None, **kwargs) -> Dict[str, T]:
        assert dataset_name is None or dataset_name == cls.HUGGINGFACE_DATASET_NAME
        ds_dict = load_dataset(cls.HUGGINGFACE_DATASET_NAME)
        return {
            "train": cls(ds_dict["train"]),
            "validate": cls(ds_dict["test"]),
        }

    def _transform_dataset(self, hf_data) -> TensorDataset:
        """
        Example of each row of hf_data:
            {'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
        """
        input_ids = []
        attention_masks = []
        labels = []
        e1_pos = []
        e2_pos = []
        actual_lens = []
        for item in hf_data:
            encoded_dict = self.tokenizer(
                item["sentence"],
                add_special_tokens=True,
                padding="max_length",
                truncation=False,
                max_length=self.MAX_SEQ_LEN,
                return_attention_mask=True,
                return_tensors="pt",
            )
            try:
                # find position of <e1> and <e2>
                ids = encoded_dict["input_ids"]
                mask = encoded_dict["attention_mask"]
                # Find e1(id:2487) and e2(id:2475) position
                pos1 = (ids == 2487).nonzero()[0][1].item()
                pos2 = (ids == 2475).nonzero()[0][1].item()
                if pos1 > self.MAX_SEQ_LEN:
                    pos1 = -1
                if pos2 > self.MAX_SEQ_LEN:
                    pos2 = -1
                # truncate manually
                if ids.shape[1] > self.MAX_SEQ_LEN:
                    ids = torch.narrow_copy(ids, 1, 0, self.MAX_SEQ_LEN)
                    ids[0, -1] = self.tokenizer.sep_token_id
                    mask = torch.narrow_copy(mask, 1, 0, self.MAX_SEQ_LEN)
                e1_pos.append(pos1)
                e2_pos.append(pos2)
                # Add the encoded sentence to the list.
                input_ids.append(ids)
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(mask)
                actual_len = torch.max(torch.arange(1, self.MAX_SEQ_LEN + 1, dtype=torch.long) * mask).item()
                labels.append(item["relation"])
                actual_lens.append(actual_len)
            except:
                pass

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        e1_pos = torch.tensor(e1_pos, dtype=torch.long)
        e2_pos = torch.tensor(e2_pos, dtype=torch.long)
        actual_lens = torch.tensor(actual_lens, dtype=torch.long)

        # Combine the training inputs into a TensorDataset.
        return TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens)
