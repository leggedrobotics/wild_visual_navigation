from wild_visual_navigation.model import get_model
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from wild_visual_navigation.visu import LearningVisualizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

# from torch_geometric.data import Data
from torchmetrics import ROC

from wild_visual_navigation.utils import TraversabilityLoss, MetricLogger
import os


class LightningTrav(pl.LightningModule):
    def __init__(self, exp: dict, log: bool = False):
        super().__init__()
        self._model = get_model(exp["model"])

        self._visu_count = {"val": 0, "test": 0, "train": 0}
        self._visualizer = LearningVisualizer(**exp["visu"]["learning_visu"], pl_model=self)
        self._exp = exp
        self._mode = "train"
        self._log = log

        self._metric_logger = MetricLogger(self.log, False)
        self._metric_logger.to(self.device)
        self._auxiliary_training_roc = ROC(task="binary")

        # The Accumulated results are used to store the results of the validation dataloader for every validation epoch. Allows after training to have summary avaiable.
        self.accumulated_val_results = []
        # The Accumulated results are used to store the results of the test dataloader. Allows after testing to have summary avaiable.
        self.accumulated_test_results = []

        self.nr_test_run = -1
        self._val_step = 0

        self._traversability_loss = TraversabilityLoss(**self._exp["loss"], model=self._model)
        threshold = torch.tensor([0.5], dtype=torch.float32, requires_grad=False)
        self.register_buffer("threshold", threshold)

    def forward(self, data: torch.tensor):
        return self._model(data)

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visualizer.epoch = self.current_epoch
        # self._auxiliary_training_roc.reset()

    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        graph = batch
        self._mode = "train"
        nr = self._exp["general"]["store_model_every_n_steps"]
        if type(nr) == int:
            if self.global_step % nr == 0:
                path = os.path.join(
                    self._exp["general"]["model_path"],
                    self._exp["general"]["store_model_every_n_steps_key"] + f"_{self.global_step}.pt",
                )
                self.update_threshold()
                torch.save(self.state_dict(), path)

        BS = graph.ptr.numel() - 1

        res = self._model(graph)
        loss, loss_aux, res_updated = self._traversability_loss(graph, res)

        for k, v in loss_aux.items():
            if k.find("loss") != -1:
                self.log(
                    f"{self._mode}_{k}",
                    v.item(),
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=BS,
                )
        self.log(
            f"{self._mode}_loss",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            batch_size=BS,
        )

        self.visu(graph, res_updated, loss_aux["confidence"])

        # This mask should contain all the segments corrosponding to trees.
        mask_anomaly = loss_aux["confidence"] < 0.5
        mask_supervision = graph.y == 1
        # Remove the segments that are for sure not an anomalies given that we have walked on them.
        mask_anomaly[mask_supervision] = False
        # Elements are valid if they are either an anomaly or we have walked on them to fit the ROC
        mask_valid = mask_anomaly | mask_supervision
        self._auxiliary_training_roc(res_updated[mask_valid, 0], graph.y[mask_valid].type(torch.long))

        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        # Log epoch metric
        self._mode = "train"
        self._metric_logger.reset("train")

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0
        self._metric_logger.reset("val")
        self._val_step = 0

    def validation_step(self, batch: any, batch_idx: int, dataloader_id: int = 0) -> torch.Tensor:
        self._mode = "val"
        graph = batch
        BS = graph.ptr.numel() - 1
        res = self._model(graph)
        loss, loss_aux, res_updated = self._traversability_loss(graph, res)

        if hasattr(graph, "label"):
            self.update_threshold()
            self._metric_logger.log_image(graph, res_updated, self._mode, threshold=self.threshold[0].item())
            self._metric_logger.log_confidence(graph, res_updated, loss_aux["confidence"], self._mode)

        for k, v in loss_aux.items():
            if k.find("loss") != -1:
                self.log(
                    f"{self._mode}_{k}",
                    v.item(),
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=BS,
                )

        self.log(
            f"{self._mode}_loss",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            batch_size=BS,
        )
        self.visu(graph, res_updated, loss_aux["confidence"])

        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        if self._exp["ablation_data_module"]["val_equals_test"]:
            res = self._metric_logger.get_epoch_results("val")
            self.accumulated_val_results.append(res)

        self._metric_logger.reset("val")

    def update_threshold(self):
        # TESTING
        if self._exp["general"]["use_threshold"]:
            try:
                fpr, tpr, thresholds = self._auxiliary_training_roc.compute()
                index = torch.where(fpr > 0.15)[0][0]
                self.threshold[0] = thresholds[index]
            except Exception:
                pass
        else:
            self.threshold[0] = 0.5

    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0
        self._metric_logger.reset("test")
        self.update_threshold()

    def test_step(self, batch: any, batch_idx: int, dataloader_id: int = 0) -> torch.Tensor:
        self._mode = "test"
        graph = batch
        BS = graph.ptr.numel() - 1
        res = self._model(graph)
        loss, loss_aux, res_updated = self._traversability_loss(graph, res)

        if hasattr(graph, "label"):
            self.update_threshold()
            self._metric_logger.log_image(graph, res_updated, self._mode, threshold=self.threshold.item())
            self._metric_logger.log_confidence(graph, res_updated, loss_aux["confidence"], self._mode)

        for k, v in loss_aux.items():
            if k.find("loss") != -1:
                self.log(
                    f"{self._mode}_{k}",
                    v.item(),
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=BS,
                )
        self.log(
            f"{self._mode}_loss",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            batch_size=BS,
        )
        self.visu(graph, res_updated, loss_aux["confidence"])
        return loss

    def test_epoch_end(self, outputs: any, plot=False):
        # NEW VERSION
        res = self._metric_logger.get_epoch_results("test")
        dic2 = {
            "trainer_logged_metric" + k: v.item()
            for k, v in self.trainer.logged_metrics.items()
            if k.find("step") == -1
        }
        res.update(dic2)

        self.accumulated_test_results.append(res)
        self._metric_logger.reset("test")
        return {}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self._exp["optimizer"]["name"] == "ADAM":
            return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
        elif self._exp["optimizer"]["name"] == "SGD":
            return torch.optim.SGD(self._model.parameters(), lr=self._exp["optimizer"]["lr"])

    def visu(self, graph: Data, res: torch.tensor, confidence: torch.tensor):
        try:
            self.log(
                f"{self._mode}_reconstruction_loss_mean_fit_per_batch",
                self._traversability_loss._confidence_generator.mean.item(),
                on_step=True,
                on_epoch=True,
                batch_size=graph.ptr.shape[0] - 1,
            )
            self.log(
                f"{self._mode}_reconstruction_loss_std_fit_per_batch",
                self._traversability_loss._confidence_generator.std.item(),
                on_step=True,
                on_epoch=True,
                batch_size=graph.ptr.shape[0] - 1,
            )
        except Exception:
            pass

        for b in range(graph.ptr.shape[0] - 1):
            log_video = self._exp["visu"][f"log_{self._mode}_video"]
            if self._mode == "val":
                self._val_step += 1

            if not log_video:
                # check if visualization should be skipped
                if not (
                    self._visu_count[self._mode] < self._exp["visu"][self._mode]
                    and self.current_epoch % self._exp["visu"]["log_every_n_epochs"] == 0
                ):
                    break

                if self._mode == "val" and self._val_step % 20 != 0:
                    continue

            self._visu_count[self._mode] += 1

            pred = res[graph.ptr[b] : graph.ptr[b + 1]]
            center = graph.center[graph.ptr[b] : graph.ptr[b + 1]]

            c = self._visu_count[self._mode]
            c = "0" * int(6 - len(str(c))) + str(c)

            if hasattr(graph, "img"):
                seg = graph.seg[b]
                img = graph.img[b]

                # Visualize Graph with Segmentation
                t1 = self._visualizer.plot_traversability_graph_on_seg(
                    pred[:, 0],
                    seg,
                    graph[b],
                    center,
                    img,
                    not_log=True,
                    colorize_invalid_centers=True,
                )
                t2 = self._visualizer.plot_traversability_graph_on_seg(
                    graph[b].y, seg, graph[b], center, img, not_log=True
                )
                t_img = self._visualizer.plot_image(img, not_log=True)
                self._visualizer.plot_list(
                    imgs=[t1, t2, t_img],
                    tag=f"C{c}_{self._mode}_GraphTrav",
                    store_folder=f"{self._mode}/graph_trav",
                )

                nr_channel_reco = graph[b].x.shape[1]
                reco_loss = F.mse_loss(pred[:, -nr_channel_reco:], graph[b].x, reduction="none").mean(dim=1)
                conf = self._traversability_loss._confidence_generator.inference_without_update(reco_loss)

                # # Visualize Graph with Segmentation
                c1 = self._visualizer.plot_traversability_graph_on_seg(
                    conf,
                    seg,
                    graph[b],
                    center,
                    img,
                    not_log=True,
                    colorize_invalid_centers=True,
                )
                c2 = self._visualizer.plot_traversability_graph_on_seg(
                    graph[b].y_valid.type(torch.float32),
                    seg,
                    graph[b],
                    center,
                    img,
                    not_log=True,
                )
                c_img = self._visualizer.plot_image(img, not_log=True)
                self._visualizer.plot_list(
                    imgs=[c1, c2, c_img],
                    tag=f"C{c}_{self._mode}_GraphConf",
                    store_folder=f"{self._mode}/graph_conf",
                )

                # if self._mode == "test":
                #     t_gt = self._visualizer.plot_traversability_graph_on_seg(
                #         graph[b].y_gt, seg, graph[b], center, img, not_log=True, colorize_invalid_centers=True
                #     )
                #     self._visualizer.plot_list(
                #         imgs=[t1, t2, t_gt, t_img],
                #         tag=f"C{c}_{self._mode}_GraphTravGT",
                #         store_folder=f"{self._mode}/graph_trav_gt",
                #     )

                buffer_confidence = graph.seg.clone().type(torch.float32).flatten()
                BS, H, W = graph.seg.shape
                # Use the position within the batch to offset the segmentation
                # This avoids iterating over the batch dimension
                batch_pixel_index_offset = graph.ptr[:-1, None, None].repeat(1, H, W)
                seg_pixel_index = (graph.seg + batch_pixel_index_offset).flatten()
                buffer_confidence = confidence[seg_pixel_index].reshape(BS, H, W)

                self._visualizer.plot_detectron_classification(img, buffer_confidence[b], tag="Confidence MAP")

                if self._mode == "val":
                    buffer_trav = graph.seg.clone().type(torch.float32).flatten()
                    buffer_trav = res[seg_pixel_index, 0].reshape(BS, H, W)

                    self._visualizer.plot_detectron_classification(
                        img,
                        buffer_trav[b],
                        tag=f"C{c}_{self._mode}_Traversability",
                    )

                    # Compute the threshold for traversability scaling
                    print("Computing threshold")
                    fpr, tpr, thresholds = self._auxiliary_training_roc.compute()
                    index = torch.where(fpr > 0.2)[0][0]
                    threshold = thresholds[index]

                    # Apply pisewise linear scaling 0->0; threshold->0.5; 1->1
                    scaled_traversability = buffer_trav.clone()
                    scale1 = 0.5 / threshold
                    m = scaled_traversability < threshold
                    scaled_traversability[m] *= scale1
                    scaled_traversability[~m] -= threshold
                    scaled_traversability[~m] *= 0.5 / (1 - threshold)
                    scaled_traversability[~m] += 0.5
                    scaled_traversability.clip(0, 1)
                    self._visualizer.plot_detectron_classification(
                        img,
                        scaled_traversability[b],
                        tag=f"C{c}_{self._mode}_Scaled_Traversability",
                    )

            # Logging the confidence
            mean = self._traversability_loss._confidence_generator.mean.item()
            std = self._traversability_loss._confidence_generator.std.item()
            nr_channel_reco = graph[b].x.shape[1]
            reco_loss = F.mse_loss(pred[:, -nr_channel_reco:], graph[b].x, reduction="none").mean(dim=1)

            self._visualizer.plot_histogram(
                reco_loss,
                graph[b].y,
                mean,
                std,
                tag=f"C{c}_{self._mode}__confidence_generator_prop",
            )

            if hasattr(graph[b], "y_gt"):
                self._visualizer.plot_histogram(
                    reco_loss,
                    graph[b].y_gt,
                    mean,
                    std,
                    tag=f"C{c}_{self._mode}__confidence_generator_gt",
                )
