from .gcn import GCN

REGISTERED_GNN_CLASSES = dict()


def register_gnn_class(cls, name):
    REGISTERED_GNN_CLASSES[name] = cls


def get_gnn_class(name: str):
    ret = REGISTERED_GNN_CLASSES.get(name)
    assert ret is not None, f"Could not find registered gnn class: {name}"
    return ret


register_gnn_class(GCN, "gcn")
