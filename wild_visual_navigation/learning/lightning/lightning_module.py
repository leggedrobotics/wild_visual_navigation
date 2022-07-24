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
        # compute loss only for valid elements [graph.y_valid]
        loss_trav = F.mse_loss(F.sigmoid(res[:, 0]), graph.y)

        loss = loss_trav * self._exp["loss"]["trav"]
        self.log(f"{self._mode}_loss_trav", loss_trav.item(), on_epoch=True, prog_bar=True)

        if self._exp["model"]["simple_gcn_cfg"]["reconstruction"]:
            nc = self._exp["model"]["simple_gcn_cfg"]["num_classes"]
            loss_reco = F.mse_loss(res[graph.y_valid][:, nc:], graph.x[graph.y_valid])
            loss += loss_reco * self._exp["loss"]["reco"]
            self.log(f"{self._mode}_loss_reco", loss_reco.item(), on_epoch=True, prog_bar=True)

        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True)

        if not fast:
            self.visu(graph, center, img, seg, res)

        return loss

    def visu(self, graph, center, img, seg, res):
        for b in range(img.shape[0]):
            if not (
                self._visu_count[self._mode] < self._exp["visu"][self._mode]
                and self.current_epoch % self._exp["visu"]["log_every_n_epochs"] == 0
            ):
                break

            self._visu_count[self._mode] += 1

            n = int(res.shape[0] / img.shape[0])
            pred = res[int(n * b) : int(n * (b + 1))]
            c = self._visu_count[self._mode]
            e = self.current_epoch
            self._visualizer.plot_graph_result(
                graph[b], center[b], img[b], seg[b], pred[:, 0], colormap="autumn", tag=f"C{c}_{self._mode}_TRAV"
            )
            nc = self._exp["model"]["simple_gcn_cfg"]["num_classes"]

            conf = (F.sigmoid(pred[:, nc:]) - graph[b].x).abs().sum(dim=1)
            conf -= conf.min()
            conf /= conf.max()
            conf = 1 - conf
            self._visualizer.plot_graph_result(
                graph[b], center[b], img[b], seg[b], conf, colormap="autumn", tag=f"C{c}_{self._mode}_CONV"
            )

    def training_epoch_end(self, outputs):
        # Log epoch metric
        self._mode = "train"

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    def validation_step(self, batch, batch_idx):
        # LAZY implementation of validation and test by calling the training method
        # Usually you want to have a different behaviour
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        pass

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0

    def test_step(self, batch, batch_idx: int) -> None:
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
