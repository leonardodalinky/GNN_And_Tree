from typing import Type

from .base import ModelBase
from .re_normal import ReNormal

REGISTERED_FRAMEWORK_CLASSES = dict()


def register_framework_class(name: str, cls):
    assert issubclass(cls, ModelBase) and cls is not ModelBase
    REGISTERED_FRAMEWORK_CLASSES[name] = cls


def get_framework_class(name: str) -> Type[ModelBase]:
    ret = REGISTERED_FRAMEWORK_CLASSES.get(name)
    assert ret is not None, f"Could not find registered framework class: {name}"
    return ret


register_framework_class(ReNormal.FRAMEWORK_NAME, ReNormal)
