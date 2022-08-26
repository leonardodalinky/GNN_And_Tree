import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
import torch.functional as F
import torch.utils.data
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from torch_geometric.data import Data, Batch
from pytorch_lightning.utilities.seed import seed_everything

from data.re import DatasetForRE
from model.gcn import GCN
from tree.left_tree import LeftTree

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)

SRC_DIR = os.path.realpath(f"{__file__}/..")
ROOT_DIR = os.path.realpath(f"{SRC_DIR}/..")
PROJECT_NAME = "test"


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256, workers=4):
        super(DataModule, self).__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage) -> None:
        if stage != "test":
            datasets = DatasetForRE.load("sem_eval_2010_task_8")

        if stage in (None, "fit"):
            self.train_dataset = datasets["train"]

        if stage in (None, "fit", "validate"):
            self.val_dataset = datasets["test"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )


class Model(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=5e-4,
    ):
        super(Model, self).__init__()
        # hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay

        self.tree = LeftTree()
        self.gcn = GCN()
        self.classifier = nn.Linear(768, 19)
        self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.train_acc = tm.Accuracy()
        self.train_f1 = tm.F1Score(average="micro")
        self.val_acc = tm.Accuracy()
        self.val_f1 = tm.F1Score(average="micro")

        self.save_hyperparameters()

    def forward(self, *args):
        # x: (batch_size, seq_len)
        batch: Batch
        embeds, edges, batch = self.tree(
            *args, task_type="re"
        )  # embeds: (batch_size, seq_len, hidden_size), edges: list(2, edges_num), Batch

        x = self.gcn(batch.x, batch.edge_index)  # x: (batch_seq_len, hidden_size)
        x = torch.reshape(x, embeds.shape)  # x: (batch, seq_len, hidden_size)
        x = torch.mean(x, dim=1)  # x: (batch_size, hidden_size)
        x = self.classifier(x)  # x: (batch_size, 19)
        return x

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens = batch
        y_hat = self.forward(input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens)
        loss = self.criterion(y_hat, labels)

        # metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        self.train_acc(y_hat, labels)
        self.log("train_acc", self.train_acc, on_epoch=True)

        self.train_f1(y_hat, labels)
        self.log("train_f1", self.train_f1, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens = batch
        y_hat = self.forward(input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens)

        self.val_acc(y_hat, labels)
        self.log("val_acc", self.val_acc, on_epoch=True)

        self.val_f1(y_hat, labels)
        self.log("val_f1", self.val_f1, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            betas=(0.9, 0.999),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": lr_scheduler,
        }


# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, data in enumerate(train_loader):
#         input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens = (item.to(device) for item in data)
#         optimizer.zero_grad()
#         output = model(input_ids, attention_masks, labels, e1_pos, e2_pos, actual_lens)
#         loss = F.cross_entropy(output, labels)
#         loss.backward()
#         optimizer.step()
#         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             epoch, batch_idx * len(data), len(train_loader.dataset),
#             100. * batch_idx / len(train_loader), loss.item()))


def main():
    seed_everything(42)

    version = "TEST"

    data_module = DataModule(
        batch_size=32,
    )

    callbacks = [
        pl_callbacks.LearningRateMonitor(),
        pl_callbacks.ModelCheckpoint(
            dirpath=os.path.realpath(f"{ROOT_DIR}/checkpoints/{PROJECT_NAME}"),
            filename=f"{version}-{PROJECT_NAME}" + "-{epoch}-{val_acc:.2f}",
            monitor="val_acc",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
    ]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(ROOT_DIR, "tb_logs"),
        name=PROJECT_NAME,
        version=f"train-{version}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # strategy="ddp",
        logger=tb_logger,
        max_epochs=20,
        callbacks=callbacks,
    )

    model = Model()
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
