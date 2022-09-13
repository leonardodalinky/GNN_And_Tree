from typing import Type, Optional

from .gat import GAT
from .gcn import GCN
from .gin import GIN
from .base import GNNBase
from .gcn_dropout import GCN_Dropout

REGISTERED_GNN_CLASSES = dict()


def register_gnn_class(cls, name: Optional[str] = None):
    assert issubclass(cls, GNNBase) and cls is not GNNBase
    name = name or cls.GNN_NAME
    assert name is not None
    REGISTERED_GNN_CLASSES[name] = cls


def get_gnn_class(name: str) -> Type[GNNBase]:
    ret = REGISTERED_GNN_CLASSES.get(name)
    assert ret is not None, f"Could not find registered gnn class: {name}"
    return ret


register_gnn_class(GCN)
register_gnn_class(GCN_Dropout)
register_gnn_class(GAT)
register_gnn_class(GIN)
