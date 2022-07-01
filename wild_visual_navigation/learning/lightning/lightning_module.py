from wild_visual_navigation.learning.model import get_model

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy 

class LightningTrav(pl.LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self._model = get_model(exp["model"])

        self.acc_val = Accuracy()
        self.acc_test = Accuracy()
        self.acc_train = Accuracy()
        self._acc = {"val": self.acc_val, "test": self.acc_test, "train": self.acc_train}

        self._visu_count = {"val": 0, "test": 0, "train": 0}

        self._exp = exp
        self._env = env
        self._mode = "train"

    def forward(self, data):
        return self._model(data)

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0

    def training_step(self, batch, batch_idx):
        res = self._model(batch)
        loss = cross_entropy(res, batch.y, reduction="mean")
        self.log(f"{self._mode}_loss", loss.item())
        preds = torch.argmax(res, dim=1)
        self._acc[self._mode](preds, batch.y)
        return loss

    def training_epoch_end(self, outputs):
        # Log epoch metric
        self.log(f"{self._mode}_acc_epoch", self._acc[self._mode], on_epoch=True, prog_bar=True)

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    def validation_step(self, batch, batch_idx):
        # LAZY implementation of validation and test by calling the training method
        # Usually you want to have a different behaviour
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.training_epoch_end(outputs)

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0

    def test_step(self, batch, batch_idx: int) -> None:
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.training_epoch_end(outputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
