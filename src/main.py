import os
import logging
import argparse

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from torch_geometric.data import Batch
from pytorch_lightning.utilities.seed import seed_everything

from gnn.gcn import GCN
from framework import get_framework_class
from datamodule import DataModule
from tree.left_tree import LeftTree

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)

SRC_DIR = os.path.realpath(f"{__file__}/..")
ROOT_DIR = os.path.realpath(f"{SRC_DIR}/..")
PROJECT_NAME = "test"

parser = argparse.ArgumentParser(description="Graph Neural Network Experiments")
parser.add_argument(
    "-c",
    "--config",
    type=argparse.FileType("r", encoding="utf-8"),
    metavar="PATH",
    required=True,
)
parser.add_argument(
    "-v",
    "--project-version",
    type=str,
    metavar="VER",
    required=True,
)
parser.add_argument(
    "-n",
    "--project-name",
    type=str,
    metavar="NAME",
    required=True,
)
parser.add_argument("-g", "--gpus", type=int, metavar="GPU_NUM", default=1, help="Number of GPUs.")
parser.add_argument(
    "-s", "--stage", type=str, metavar="STAGE", required=True, help="Should be `fit`, `validate` or `test`."
)


def main():
    seed_everything(42)

    args = parser.parse_args()

    project_version = args.project_version
    project_name = args.project_name

    config = yaml.safe_load(args.config)

    data_module = DataModule(config)

    model = get_framework_class(config["model"]["framework"])(config)

    callbacks = [
        pl_callbacks.LearningRateMonitor(),
        pl_callbacks.ModelCheckpoint(
            dirpath=os.path.realpath(f"{ROOT_DIR}/checkpoints/{project_name}"),
            filename=f"{project_name}-{project_version}" + "-{epoch}-{val_acc:.2f}",
            monitor="val_acc",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
    ]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(ROOT_DIR, "tb_logs"),
        name=project_name,
        version=f"train-{project_version}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        # strategy="ddp",
        logger=tb_logger,
        max_epochs=config["task"]["train"].get("epochs", 30),
        callbacks=callbacks,
    )

    assert args.stage in ("train", "validate", "test")
    if args.stage == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        raise NotImplementedError(f"Stage {args.stage} is not implemented yet.")


if __name__ == "__main__":
    main()
