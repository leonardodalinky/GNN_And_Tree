from typing import TypeVar

import pytorch_lightning as pl

T = TypeVar("T")


class ModelBase(pl.LightningModule):
    FRAMEWORK_NAME = "model_base"

    def __init__(self, config):
        pass
