from typing import Type, Optional

from .base import TreeBase
from .left_tree import LeftTree
from .right_tree import RightTree

REGISTERED_TREE_CLASSES = dict()


def register_tree_class(cls, name: Optional[str] = None):
    assert issubclass(cls, TreeBase) and cls is not TreeBase
    name = name or cls.TREE_NAME
    assert name is not None
    REGISTERED_TREE_CLASSES[name] = cls


def get_tree_class(name: str) -> Type[TreeBase]:
    ret = REGISTERED_TREE_CLASSES.get(name)
    assert ret is not None, f"Could not find registered tree class: {name}"
    return ret


register_tree_class(LeftTree)
register_tree_class(RightTree)
