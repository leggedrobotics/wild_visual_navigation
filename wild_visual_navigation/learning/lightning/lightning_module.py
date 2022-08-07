from wild_visual_navigation.learning.model import get_model
from wild_visual_navigation.learning.utils import get_confidence, compute_loss

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from os.path import join
from wild_visual_navigation.visu import LearningVisualizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.data import Data


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
        graph = batch[0]
        graph_aux = batch[1]

        res = self._model(graph)

        loss, loss_aux = compute_loss(graph, res, self._exp["loss"], self._model, graph_aux)

        self.log(f"{self._mode}_loss_trav", loss_aux["loss_trav"].item(), on_epoch=True, prog_bar=True)
        self.log(f"{self._mode}_loss_reco", loss_aux["loss_reco"].item(), on_epoch=True, prog_bar=True)
        self.log(f"{self._mode}_loss_temp", loss_aux["loss_temp"].item(), on_epoch=True, prog_bar=True)
        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True)

        self.visu(graph, res)
        return loss

    def visu(self, graph: Data, res: torch.tensor):
        if not hasattr(graph, "img"):
            return

        for b in range(graph.img.shape[0]):
            if not (
                self._visu_count[self._mode] < self._exp["visu"][self._mode]
                and self.current_epoch % self._exp["visu"]["log_every_n_epochs"] == 0
            ):
                break

            self._visu_count[self._mode] += 1

            pred = res[graph.ptr[b] : graph.ptr[b + 1]]
            center = graph.center[graph.ptr[b] : graph.ptr[b + 1]]
            seg = graph.seg[b]
            img = graph.img[b]

            c = self._visu_count[self._mode]
            e = self.current_epoch

            # Visualize Graph without Segmentation
            # i1 = self._visualizer.plot_traversability_graph(pred[:, 0], graph[b], center[b], img[b], not_log=True)
            # i2 = self._visualizer.plot_traversability_graph(graph[b].y, graph[b], center[b], img[b], not_log=True)
            # self._visualizer.plot_list(imgs=[i1, i2], tag=f"C{c}_{self._mode}_Trav")

            # Visualize Graph with Segmentation
            i1 = self._visualizer.plot_traversability_graph_on_seg(pred[:, 0], seg, graph[b], center, img, not_log=True)
            i2 = self._visualizer.plot_traversability_graph_on_seg(graph[b].y, seg, graph[b], center, img, not_log=True)
            i3 = self._visualizer.plot_image(img, not_log=True)
            self._visualizer.plot_list(imgs=[i1, i2, i3], tag=f"C{c}_{self._mode}_GraphTrav")

            conf = get_confidence(pred[:, 1:], graph[b].x)
            # Visualize Graph with Segmentation
            i1 = self._visualizer.plot_traversability_graph_on_seg(conf, seg, graph[b], center, img, not_log=True)
            i2 = self._visualizer.plot_traversability_graph_on_seg(
                graph[b].y_valid.type(torch.float32), seg, graph[b], center, img, not_log=True
            )
            i3 = self._visualizer.plot_image(img, not_log=True)
            self._visualizer.plot_list(imgs=[i1, i2, i3], tag=f"C{c}_{self._mode}_GraphConf")

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
