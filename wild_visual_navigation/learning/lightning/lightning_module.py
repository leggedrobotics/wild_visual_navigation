from wild_visual_navigation.learning.model import get_model

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from os.path import join
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from wild_visual_navigation.visu import LearningVisualizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch


class LightningTrav(pl.LightningModule):
    def __init__(self, exp: dict, env: dict, log: bool = False):
        super().__init__()
        self._model = get_model(exp["model"])

        self._visu_count = {"val": 0, "test": 0, "train": 0}
        self._visualizer = LearningVisualizer(
            p_visu=join(exp["general"]["model_path"], "visu"), store=True, pl_model=self, log=True
        )
        self._exp = exp
        self._env = env
        self._mode = "train"
        self._log = log

    def forward(self, data: torch.tensor):
        return self._model(data)

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visualizer.epoch = self.current_epoch

    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
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
        res[:, 0] = F.sigmoid(res[:, 0])

        loss_trav = F.mse_loss(res[:, 0], graph.y)

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

    def visu(
        self, graph: torch_geometric.Data, center: torch.tensor, img: torch.tensor, seg: torch.tensor, res: torch.tensor
    ):
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

            # Visualize Graph without Segmentation
            i1 = self._visualizer.plot_traversability_graph(pred[:, 0], graph[b], center[b], img[b], not_log=True)
            i2 = self._visualizer.plot_traversability_graph(graph[b].y, graph[b], center[b], img[b], not_log=True)
            self._visualizer.plot_list(imgs=[i1, i2], tag=f"C{c}_{self._mode}_TRAV")

            # Visualize Graph with Segmentation
            i1 = self._visualizer.plot_traversability_graph_on_seg(
                pred[:, 0], seg[b], graph[b], center[b], img[b], not_log=True
            )
            i2 = self._visualizer.plot_traversability_graph_on_seg(
                graph[b].y, seg[b], graph[b], center[b], img[b], not_log=True
            )
            i3 = self._visualizer.plot_image(img[b], not_log=True)
            self._visualizer.plot_list(imgs=[i1, i2, i3], tag=f"C{c}_{self._mode}_GraphTRAV")

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        # Log epoch metric
        self._mode = "train"

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    def validation_step(self, batch: any, batch_idx: int):
        # LAZY implementation of validation and test by calling the training method
        # Usually you want to have a different behaviour
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        pass

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0

    def test_step(self, batch: any, batch_idx: int) -> None:
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs: any):
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
