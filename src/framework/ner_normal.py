from typing import Dict

import torch
import torch.nn as nn
import torchmetrics as tm
from torch_geometric.data import Batch

from gnn import get_gnn_class
from data import DatasetForNER
from tree import get_tree_class

from .base import ModelBase


class NerNormal(ModelBase):
    FRAMEWORK_NAME = "ner_normal"
    TASK_TYPE = "ner"

    def __init__(self, config):
        super(NerNormal, self).__init__(config)
        assert config["task"]["type"] == self.TASK_TYPE
        assert config["model"]["framework"] == self.FRAMEWORK_NAME
        self.config = config

        # hyperparameters
        self.dataset_cls = DatasetForNER.load_cls(config["task"]["dataset"]["name"])
        self.classes_num = self.dataset_cls.CLASSES_NUM
        config_train = config["task"].get("train", {})
        self.lr = config_train.get("lr", 1.0e-4)
        self.weight_decay = config_train.get("weight_decay", 5.0e-4)

        # model compositions
        self.tree = get_tree_class(config["model"]["tree"])()
        self.gnn = get_gnn_class(config["model"]["gnn"])()
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.tree.EMBEDDING_SIZE, self.classes_num)

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
        # x: (batch_size, seq_len)
        batch: Batch
        embeds, edges, batch = self.tree(
            data_dict, task_type=self.TASK_TYPE
        )  # embeds: (batch_size, seq_len, hidden_size), edges: list(2, edges_num), Batch

        x = self.gnn(batch.x, batch.edge_index)  # x: (batch_seq_len, hidden_size)
        x = torch.reshape(x, embeds.shape)  # x: (batch_size, seq_len, hidden_size)
        x = self.classifier(x)  # x: (batch_size, seq_len, class_num)
        return x

    def training_step(self, batch, batch_idx):
        labels = batch["ner_tags"]  # labels: (batch_size, seq_len)
        actual_lens = batch["actual_lens"]  # actual_lens: (batch_size)
        y_hat = self.forward(batch)  # y_hat: (batch_size, seq_len, class_num)
        all_loss = torch.zeros(y_hat.shape[0], dtype=torch.float)

        for i, length in enumerate(actual_lens):
            all_loss[i] = self.criterion(y_hat[i, :length], labels[i, :length])

        loss = torch.mean(all_loss)

        # metrics
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        for i, length in enumerate(actual_lens):
            y_h = y_hat[i, :length]
            y = labels[i, :length]
            self.train_acc_w.update(y_h, y)
            self.train_f1_w(y_h, y)

            if self.dataset_cls.IGNORED_CLASS_INDEX is not None:
                self.train_acc_wo(y_h, y)
                self.train_f1_wo(y_h, y)

        self.log("train_acc_w", self.train_acc_w, on_epoch=True)
        self.log("train_f1_w", self.train_f1_w, on_epoch=True)
        self.log("train_acc_wo", self.train_acc_wo, on_epoch=True)
        self.log("train_f1_wo", self.train_f1_wo, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["ner_tags"]  # labels: (batch_size, seq_len)
        actual_lens = batch["actual_lens"]  # actual_lens: (batch_size)
        y_hat = self.forward(batch)  # y_hat: (batch_size, seq_len, class_num)

        for i, length in enumerate(actual_lens):
            y_h = y_hat[i, :length]
            y = labels[i, :length]
            self.val_acc_w.update(y_h, y)
            self.val_f1_w(y_h, y)

            if self.dataset_cls.IGNORED_CLASS_INDEX is not None:
                self.val_acc_wo(y_h, y)
                self.val_f1_wo(y_h, y)

        self.log("val_acc_w", self.val_acc_w, on_epoch=True)
        self.log("val_f1_w", self.val_f1_w, on_epoch=True)
        self.log("val_acc_wo", self.val_acc_wo, on_epoch=True)
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
