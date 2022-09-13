import sys
from typing import Dict

import torch
import torch.nn as nn
import torchmetrics as tm
from torch_geometric.data import Batch

from gnn import get_gnn_class
from data import DatasetForRE
from tree import get_tree_class

from .base import ModelBase


class RePos(ModelBase):
    FRAMEWORK_NAME = "re_pos"
    TASK_TYPE = "re"

    def __init__(self, config):
        super().__init__(config)
        assert config["task"]["type"] == self.TASK_TYPE
        assert config["model"]["framework"] == self.FRAMEWORK_NAME
        self.config = config

        # hyperparameters
        self.dataset_cls = DatasetForRE.load_cls(config["task"]["dataset"]["name"])
        self.classes_num = self.dataset_cls.CLASSES_NUM
        config_train = config["task"].get("train", {})
        self.lr = config_train.get("lr", 1.0e-4)
        self.weight_decay = config_train.get("weight_decay", 5.0e-4)

        # model compositions
        self.tree = get_tree_class(config["model"]["tree"])()
        self.gnn = get_gnn_class(config["model"]["gnn"])()
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.tree.EMBEDDING_SIZE, 2 * self.tree.EMBEDDING_SIZE // 4),
            nn.SiLU(),
            nn.Linear(2 * self.tree.EMBEDDING_SIZE // 4, self.classes_num),
        )

        # metrics

        self.train_acc_w = tm.Accuracy()
        self.train_f1_w = tm.F1Score(average="micro")
        self.val_acc_w = tm.Accuracy()
        self.val_f1_w = tm.F1Score(average="micro")
        if self.dataset_cls.IGNORED_CLASS_INDEX is not None:
            self.train_acc_wo = tm.Accuracy(ignore_index=self.dataset_cls.IGNORED_CLASS_INDEX)
            self.train_f1_wo = tm.F1Score(average="micro", ignore_index=self.dataset_cls.IGNORED_CLASS_INDEX)
            self.val_acc_wo = tm.Accuracy(ignore_index=self.dataset_cls.IGNORED_CLASS_INDEX)
            self.val_f1_wo = tm.F1Score(average="micro", ignore_index=self.dataset_cls.IGNORED_CLASS_INDEX)

        self.save_hyperparameters()

    def forward(self, data_dict: Dict[str, torch.Tensor]):
        e1_pos = data_dict["e1_pos"]
        e2_pos = data_dict["e2_pos"]
        batch: Batch
        embeds, edges, batch = self.tree(
            data_dict, task_type=self.TASK_TYPE
        )  # embeds: (batch_size, seq_len, hidden_size), edges: list(2, edges_num), Batch

        if torch.isnan(batch.x).sum().item() != 0:
            print("batch.x is nan")
            sys.exit(1)

        x = self.gnn(batch.x, batch.edge_index)  # x: (batch_seq_len, hidden_size)
        if torch.isnan(x).sum().item() != 0:
            print("gnn x is nan")
            sys.exit(1)
        x = torch.reshape(x, embeds.shape)  # x: (batch, seq_len, hidden_size)
        cat_pos_tensors = []
        for xx, e1, e2 in zip(x, e1_pos, e2_pos):
            # xx: (seq_len, hidden_size)
            cat_pos_tensors.append(torch.cat([xx[e1], xx[e2]], dim=0))
        cat_pos_tensors = torch.stack(cat_pos_tensors, dim=0)  # (batch, 2 * hidden_size)
        ret = self.classifier(cat_pos_tensors)  # (batch, class_num)
        if torch.isnan(ret).sum().item() != 0:
            print("ret is nan")
            sys.exit(1)
        return ret

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        y_hat = self.forward(batch)
        loss = self.criterion(y_hat, labels)
        if torch.isnan(loss).sum().item() != 0:
            print("loss is nan")
            sys.exit(1)

        # metrics
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        self.train_acc_w(y_hat, labels)
        self.log("train_acc_w", self.train_acc_w, on_epoch=True)
        self.train_f1_w(y_hat, labels)
        self.log("train_f1_w", self.train_f1_w, on_epoch=True)

        if self.dataset_cls.IGNORED_CLASS_INDEX is not None:
            self.train_acc_wo(y_hat, labels)
            self.log("train_acc_wo", self.train_acc_wo, on_epoch=True)
            self.train_f1_wo(y_hat, labels)
            self.log("train_f1_wo", self.train_f1_wo, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        y_hat = self.forward(batch)

        self.val_acc_w(y_hat, labels)
        self.log("val_acc_w", self.val_acc_w, on_epoch=True)
        self.val_f1_w(y_hat, labels)
        self.log("val_f1_w", self.val_f1_w, on_epoch=True)

        if self.dataset_cls.IGNORED_CLASS_INDEX is not None:
            self.val_acc_wo(y_hat, labels)
            self.log("val_acc_wo", self.val_acc_wo, on_epoch=True)
            self.val_f1_wo(y_hat, labels)
            self.log("val_f1_wo", self.val_f1_wo, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # TODO
        pass

    def configure_optimizers(self):
        epochs = self.config["task"]["train"].get("epochs", 30)
        optimizer = torch.optim.AdamW(
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
