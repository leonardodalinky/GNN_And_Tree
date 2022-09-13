from typing import Type, Optional

from .base import ModelBase
from .re_pos import RePos
from .re_normal import ReNormal
from .ner_normal import NerNormal
from .re_normal_v2 import ReNormal_v2
from .ner_normal_v2 import NerNormal_v2

REGISTERED_FRAMEWORK_CLASSES = dict()


def register_framework_class(cls, name: Optional[str] = None):
    assert issubclass(cls, ModelBase) and cls is not ModelBase
    name = name or cls.FRAMEWORK_NAME
    assert name is not None
    REGISTERED_FRAMEWORK_CLASSES[name] = cls


def get_framework_class(name: str) -> Type[ModelBase]:
    ret = REGISTERED_FRAMEWORK_CLASSES.get(name)
    assert ret is not None, f"Could not find registered framework class: {name}"
    return ret


register_framework_class(ReNormal)
register_framework_class(ReNormal_v2)
register_framework_class(RePos)
register_framework_class(NerNormal)
register_framework_class(NerNormal_v2)
