from typing import Type

from .base import TreeBase
from .left_tree import LeftTree

REGISTERED_TREE_CLASSES = dict()


def register_tree_class(name: str, cls):
    assert issubclass(cls, TreeBase) and cls is not TreeBase
    REGISTERED_TREE_CLASSES[name] = cls


def get_tree_class(name: str) -> Type[TreeBase]:
    ret = REGISTERED_TREE_CLASSES.get(name)
    assert ret is not None, f"Could not find registered tree class: {name}"
    return ret


register_tree_class(LeftTree.TREE_NAME, LeftTree)
