from typing import Dict, TypeVar

import torch
from datasets import load_dataset
from transformers import BertTokenizer

from utils.data import DictTensorDataset

from .base import DatasetForNER

T = TypeVar("T")


class OntoNotesv5ForNER(DatasetForNER):
    _NER_TAGS = [
        "O",
        "B-PERSON",
        "I-PERSON",
        "B-NORP",
        "I-NORP",
        "B-FAC",
        "I-FAC",
        "B-ORG",
        "I-ORG",
        "B-GPE",
        "I-GPE",
        "B-LOC",
        "I-LOC",
        "B-PRODUCT",
        "I-PRODUCT",
        "B-DATE",
        "I-DATE",
        "B-TIME",
        "I-TIME",
        "B-PERCENT",
        "I-PERCENT",
        "B-MONEY",
        "I-MONEY",
        "B-QUANTITY",
        "I-QUANTITY",
        "B-ORDINAL",
        "I-ORDINAL",
        "B-CARDINAL",
        "I-CARDINAL",
        "B-EVENT",
        "I-EVENT",
        "B-WORK_OF_ART",
        "I-WORK_OF_ART",
        "B-LAW",
        "I-LAW",
        "B-LANGUAGE",
        "I-LANGUAGE",
    ]
    NER_TAG_2_IDX = {tag: i for i, tag in enumerate(_NER_TAGS)}
    NER_IDX_2_TAG = {v: k for k, v in NER_TAG_2_IDX.items()}
    HUGGINGFACE_DATASET_NAME = "conll2012_ontonotesv5"
    HUGGINGFACE_SUBDATASET_NAME = "english_v4"
    BERT_TOKENIZER_NAME = "bert-base-uncased"
    MAX_SEQ_LEN = 160
    CLASSES_NUM = len(NER_TAG_2_IDX)
    IGNORED_CLASS_INDEX = 0

    def __init__(self, hf_data):
        super(OntoNotesv5ForNER, self).__init__()
        self._data = self._transform_dataset(hf_data)

    @classmethod
    def get_dataset_name(cls):
        return cls.HUGGINGFACE_DATASET_NAME

    @property
    def inner_data(self):
        """
        Example of each row:
            {'part_id': 0, 'words': ['What', 'kind', 'of', 'memory', '?'], 'pos_tags': [46, 24, 17, 24, 7], 'parse_tree': '(TOP(SBARQ(WHNP(WHNP (WP What)  (NN kind) )(PP (IN of) (NP (NN memory) ))) (. ?) ))', 'predicate_lemmas': [None, None, None, 'memory', None], 'predicate_framenet_ids': [None, None, None, None, None], 'word_senses': [None, None, None, 1.0, None], 'speaker': 'Speaker#1', 'named_entities': [0, 0, 0, 0, 0], 'srl_frames': [], 'coref_spans': []}
        """
        return self._data

    @classmethod
    def load(cls: T, dataset_name=None, **kwargs) -> Dict[str, T]:
        assert dataset_name is None or dataset_name == cls.HUGGINGFACE_DATASET_NAME
        ds_dict = load_dataset(cls.HUGGINGFACE_DATASET_NAME, cls.HUGGINGFACE_SUBDATASET_NAME)
        return {
            "train": cls(ds_dict["train"]),
            "validate": cls(ds_dict["validation"]),
            "test": cls(ds_dict["test"]),
        }

    def _transform_dataset(self, hf_data):
        cls = self.__class__
        input_ids = []
        attention_masks = []
        ner_tags = []
        actual_lens = []

        # Load tokenizer.
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.BERT_TOKENIZER_NAME, do_lower_case=True)
        # pre-processing sentenses to BERT pattern
        for document in hf_data:
            for data in document["sentences"]:
                tokens, tags = list(data["words"]), list(data["named_entities"])
                # manually add special token
                tokens.insert(0, "[CLS]")
                tokens.append("[SEP]")
                tags.insert(0, cls.NER_TAG_2_IDX["O"])
                tags.append(cls.NER_TAG_2_IDX["O"])
                # padding tags manually
                if len(tags) < cls.MAX_SEQ_LEN:
                    tags.extend([cls.NER_TAG_2_IDX["O"] for i in range(cls.MAX_SEQ_LEN - len(tags))])
                # bert
                encoded_dict = tokenizer(
                    tokens,  # Sentence to encode.
                    add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                    max_length=cls.MAX_SEQ_LEN,
                    is_split_into_words=True,  # pretokenized
                    padding="max_length",
                    truncation=False,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors="pt",  # Return pytorch tensors.
                )
                try:
                    ids = encoded_dict["input_ids"]
                    mask = encoded_dict["attention_mask"]
                    # truncate manually
                    if ids.shape[1] > cls.MAX_SEQ_LEN:
                        ids = torch.narrow_copy(ids, 1, 0, cls.MAX_SEQ_LEN)
                        ids[0, -1] = tokenizer.sep_token_id
                        mask = torch.narrow_copy(mask, 1, 0, cls.MAX_SEQ_LEN)
                        tags = tags[: cls.MAX_SEQ_LEN]
                        tags[-1] = cls.NER_TAG_2_IDX["O"]
                    # Add the encoded sentence to the list.
                    input_ids.append(ids)
                    # And its attention mask (simply differentiates padding from non-padding).
                    attention_masks.append(mask)
                    ner_tags.append(tags)
                    actual_len = torch.max(torch.arange(1, cls.MAX_SEQ_LEN + 1, dtype=torch.long) * mask).item()
                    actual_lens.append(actual_len)
                except:
                    pass
                    # print(sent)

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        ner_tags = torch.tensor(ner_tags, dtype=torch.long)
        actual_lens = torch.tensor(actual_lens, dtype=torch.long)

        # Combine the training inputs into a TensorDataset.
        return DictTensorDataset(
            {
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "ner_tags": ner_tags,
                "actual_lens": actual_lens,
            }
        )
