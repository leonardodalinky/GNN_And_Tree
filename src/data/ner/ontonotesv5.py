from typing import Dict, TypeVar

from datasets import load_dataset

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

    def __init__(self, hf_data):
        super(OntoNotesv5ForNER, self).__init__()
        self._data = hf_data

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
        ret = {}
        for k, docs in ds_dict.items():
            ret[k] = []
            for doc in docs:
                ret[k] += doc["sentences"]
        return ret

    def _transform_to_task_specific_format(self, item):
        return {
            "words": item["words"],
            "ner_tags": item["named_entities"],
        }
