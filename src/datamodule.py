import torch
import torch.utils.data
import pytorch_lightning as pl

from data.ee import DatasetForEE
from data.re import DatasetForRE
from data.ner import DatasetForNER


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(DataModule, self).__init__()
        self.type = config["task"]["type"]
        assert self.type in ["re", "ner", "ee"]
        dataset_config = config["task"]["dataset"]
        self.dataset_name = dataset_config["name"]
        self.batch_size = dataset_config["batch_size"]
        self.workers = dataset_config.get("workers", 4)
        if dataset_config.get("local", False):
            raise NotImplementedError("`local` datasets is not supported now.")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage) -> None:
        # get datasets by task type and dataset name
        if self.type == "re":
            datasets = DatasetForRE.load(self.dataset_name)
        elif self.type == "ner":
            datasets = DatasetForNER.load(self.dataset_name)
        elif self.type == "ee":
            datasets = DatasetForEE.load(self.dataset_name)

        if stage in (None, "fit"):
            self.train_dataset = datasets.get("train")

        if stage in (None, "fit", "validate"):
            self.val_dataset = datasets.get("validate")

        if stage in (None, "fit", "validate", "test"):
            self.test_dataset = datasets.get("test")

    def train_dataloader(self):
        assert self.train_dataset is not None, "Couldn't get to training dataset."
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None, "Couldn't get to validation dataset."
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "Couldn't get to test dataset."
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )
