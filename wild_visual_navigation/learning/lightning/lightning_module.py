from wild_visual_navigation.learning.model import get_model

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from os.path import join
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from wild_visual_navigation.learning.visu import LearningVisualizer


class LightningTrav(pl.LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self._model = get_model(exp["model"])
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()
        self.acc_train = Accuracy()
        self._acc = {"val": self.acc_val, "test": self.acc_test, "train": self.acc_train}

        self._visu_count = {"val": 0, "test": 0, "train": 0}
        self._visualizer = LearningVisualizer(
            p_visu=join(exp["general"]["model_path"], "visu"), store=True, pl_model=self
        )
        self._exp = exp
        self._env = env
        self._mode = "train"

    def forward(self, data):
        return self._model(data)

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visualizer.epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        fast = type(batch) != list
        if fast:
            graph = batch
        else:
            graph = batch[0]
            center = batch[1]
            img = batch[2]
            seg = batch[3]

        res = self._model(graph)
        loss = cross_entropy(res, graph.y, reduction="mean")
        self.log(f"{self._mode}_loss", loss.item())
        preds = torch.argmax(res, dim=1)
        self._acc[self._mode](preds, graph.y)

        if not fast:
            self.visu(graph, center, img, seg, res)

        return loss

    def visu(self, graph, center, img, seg, res):
        if (
            self._visu_count[self._mode] < self._exp["visu"][self._mode]
            and self.current_epoch % self._exp["visu"]["log_every_n_epochs"] == 0
        ):

            r = torch.argmax(res, 1)
            for b in range(img.shape[0]):
                self._visu_count[self._mode] += 1
                n = int(res.shape[0] / img.shape[0])
                pred = r[int(n * b) : int(n * (b + 1))]
                c = self._visu_count[self._mode]
                e = self.current_epoch
                self._visualizer.plot_graph_result(
                    graph[b], center[b], img[b], seg[b], pred, tag=f"C{c}_{self._mode}_graph"
                )

    def training_epoch_end(self, outputs):
        # Log epoch metric
        self._mode = "train"
        self.log(f"{self._mode}_acc_epoch", self._acc[self._mode].compute().item(), on_epoch=True, prog_bar=True)

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    def validation_step(self, batch, batch_idx):
        # LAZY implementation of validation and test by calling the training method
        # Usually you want to have a different behaviour
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        self.log(f"{self._mode}_acc_epoch", self._acc[self._mode].compute().item(), on_epoch=True, prog_bar=True)

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0

    def test_step(self, batch, batch_idx: int) -> None:
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.log(f"{self._mode}_acc_epoch", self._acc[self._mode].compute().item(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
