import os
import logging
import argparse

import yaml
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from pytorch_lightning.utilities.seed import seed_everything

from framework import get_framework_class
from datamodule import DataModule

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=logging.INFO)

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
parser.add_argument("-g", "--gpus", type=int, metavar="GPU_NUM", default=1, help="Number of GPUs. Default: 1.")
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
            filename=f"{project_name}-{project_version}" + "-{epoch}-{val_f1_wo:.2f}",
            monitor="val_f1_wo",
            mode="max",
            # save_last=True,
            save_top_k=1,
        ),
    ]
    early_stopping = config["task"]["train"].get("early_stopping", False)
    if isinstance(early_stopping, bool) and early_stopping:
        callbacks.append(pl_callbacks.EarlyStopping("val_f1_wo", patience=10, mode="max"))
    elif isinstance(early_stopping, dict):
        callbacks.append(
            pl_callbacks.EarlyStopping("val_f1_wo", patience=early_stopping.get("patience", 10), mode="max")
        )
    else:
        raise ValueError("`early_stopping` must be bool type or dict type.")

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(ROOT_DIR, "tb_logs"),
        name=project_name,
        version=f"train-{project_version}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp",
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
