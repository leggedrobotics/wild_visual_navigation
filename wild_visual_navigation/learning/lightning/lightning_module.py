from wild_visual_navigation.learning.model import get_model
from wild_visual_navigation.learning.utils import get_confidence, compute_loss

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from os.path import join
from wild_visual_navigation.visu import LearningVisualizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.data import Data
from torchmetrics import ROC, AUROC
from wild_visual_navigation.learning.utils import TraversabilityLoss
import os
import pickle


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
        self._validation_roc_proprioceptive = ROC()
        self._validation_auroc_proprioceptive = AUROC()

        self._test_roc_proprioceptive = ROC()
        self._test_roc_gt = ROC()
        self._test_auroc_proprioceptive = AUROC()
        self._test_auroc_gt = AUROC()

        self._traversability_loss = TraversabilityLoss(**self._exp["loss"], model=self._model)

    def forward(self, data: torch.tensor):
        return self._model(data)

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visualizer.epoch = self.current_epoch
        self._traversability_loss.reset()

    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        graph = batch[0]
        graph_aux = batch[1]
        BS = graph.ptr.numel() - 1

        res = self._model(graph)
        loss, loss_aux = self._traversability_loss(graph, res, graph_aux)

        for k, v in loss_aux.items():
            self.log(f"{self._mode}_{k}", v.item(), on_epoch=True, prog_bar=True, batch_size=BS)
        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.visu(graph, res)
        return loss

    def visu(self, graph: Data, res: torch.tensor):
        if not hasattr(graph, "img"):
            return

        for b in range(graph.img.shape[0]):
            log_video = self._exp["visu"][f"log_{self._mode}_video"]

            if not log_video:
                # check if visualization should be skipped
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
            c = "0" * int(6 - len(str(c))) + str(c)
            e = self.current_epoch

            # Visualize Graph without Segmentation
            # i1 = self._visualizer.plot_traversability_graph(pred[:, 0], graph[b], center[b], img[b], not_log=True)
            # i2 = self._visualizer.plot_traversability_graph(graph[b].y, graph[b], center[b], img[b], not_log=True)
            # self._visualizer.plot_list(imgs=[i1, i2], tag=f"C{c}_{self._mode}_Trav")

            # Visualize Graph with Segmentation
            t1 = self._visualizer.plot_traversability_graph_on_seg(
                pred[:, 0], seg, graph[b], center, img, not_log=True, colorize_invalid_centers=True
            )
            t2 = self._visualizer.plot_traversability_graph_on_seg(graph[b].y, seg, graph[b], center, img, not_log=True)
            t_img = self._visualizer.plot_image(img, not_log=True)
            self._visualizer.plot_list(
                imgs=[t1, t2, t_img], tag=f"C{c}_{self._mode}_GraphTrav", store_folder=f"{self._mode}/graph_trav"
            )

            # reco_loss = loss_reco_raw = F.mse_loss(pred[:, 1:], graph[b].x, reduction="none").mean(dim=1)
            # conf = self._traversability_loss._confidence_generator.inference_without_update(reco_loss)

            # # Visualize Graph with Segmentation
            # c1 = self._visualizer.plot_traversability_graph_on_seg(
            #     conf, seg, graph[b], center, img, not_log=True, colorize_invalid_centers=True
            # )
            # c2 = self._visualizer.plot_traversability_graph_on_seg(
            #     graph[b].y_valid.type(torch.float32), seg, graph[b], center, img, not_log=True
            # )
            # c_img = self._visualizer.plot_image(img, not_log=True)
            # self._visualizer.plot_list(
            #     imgs=[c1, c2, c_img], tag=f"C{c}_{self._mode}_GraphConf", store_folder=f"{self._mode}/graph_conf"
            # )

            # if self._mode == "test":
            #     t_gt = self._visualizer.plot_traversability_graph_on_seg(
            #         graph[b].y_gt, seg, graph[b], center, img, not_log=True, colorize_invalid_centers=True
            #     )
            #     self._visualizer.plot_list(
            #         imgs=[t1, t2, t_gt, t_img],
            #         tag=f"C{c}_{self._mode}_GraphTravGT",
            #         store_folder=f"{self._mode}/graph_trav_gt",
            #     )

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        # Log epoch metric
        self._mode = "train"

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    def validation_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        graph = batch[0]
        BS = graph.ptr.numel() - 1
        graph_aux = batch[1]

        res = self._model(graph)

        loss, loss_aux = self._traversability_loss(graph, res, graph_aux)

        self._validation_roc_proprioceptive.update(preds=res[:, 0], target=(graph.y > 0).type(torch.long))
        self._validation_auroc_proprioceptive.update(res[:, 0], (graph.y > 0).type(torch.long))

        for k, v in loss_aux.items():
            self.log(f"{self._mode}_{k}", v.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.visu(graph, res)
        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        fpr, tpr, thresholds = self._validation_roc_proprioceptive.compute()
        auroc = self._validation_auroc_proprioceptive.compute().item()

        self._visualizer.plot_roc(
            x=fpr.cpu().numpy(), y=tpr.cpu().numpy(), y_tag=f"AUCROC_{auroc:.4f}", tag=f"ROC_{self._mode}"
        )

        self.log(f"{self._mode}_auroc", auroc, on_epoch=True, prog_bar=False)

        if self._exp["visu"]["log_val_video"]:
            inp = os.path.join(self._visualizer._p_visu, f"{self._mode}/graph_trav")
            out2 = os.path.join(self._visualizer._p_visu, f"{self._mode}/epoch_{self.current_epoch}_graph_trav.mp4")
            cmd = f"ffmpeg -framerate 2 -pattern_type glob -i '{inp}/{self.current_epoch}*_{self._mode}_GraphTrav.png' -c:v libx264 -pix_fmt yuv420p {out2}"
            os.system(cmd)

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0

    def test_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        graph = batch[0]
        BS = graph.ptr.numel() - 1
        graph_aux = batch[1]

        res = self._model(graph)

        loss, loss_aux = self._traversability_loss(graph, res, graph_aux)

        self._test_roc_proprioceptive.update(preds=res[:, 0], target=(graph.y > 0).type(torch.long))
        self._test_roc_gt.update(preds=res[:, 0], target=(graph.y_gt > 0).type(torch.long))
        self._test_auroc_proprioceptive.update(preds=res[:, 0], target=(graph.y > 0).type(torch.long))
        self._test_auroc_gt.update(preds=res[:, 0], target=(graph.y_gt > 0).type(torch.long))
        for k, v in loss_aux.items():
            self.log(f"{self._mode}_{k}", v.item(), on_epoch=True, prog_bar=True, batch_size=BS)
        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.visu(graph, res)

        return loss

    def test_epoch_end(self, outputs: any):
        fpr_pro, tpr_pro, thresholds_pro = self._test_roc_proprioceptive.compute()
        auroc_pro = self._test_auroc_proprioceptive.compute().item()
        self._visualizer.plot_roc(
            x=fpr_pro.cpu().numpy(),
            y=tpr_pro.cpu().numpy(),
            y_tag=f"AUCROC_{auroc_pro:.4f}",
            tag=f"{self._mode}_ROC_proprioceptive",
        )
        self.log(f"{self._mode}_auroc_proprioceptive", auroc_pro, on_epoch=True, prog_bar=False)

        fpr_gt, tpr_gt, thresholds_gt = self._test_roc_gt.compute()
        auroc_gt = self._test_auroc_gt.compute().item()
        self._visualizer.plot_roc(
            x=fpr_gt.cpu().numpy(), y=tpr_gt.cpu().numpy(), y_tag=f"AUCROC_{auroc_gt:.4f}", tag=f"{self._mode}_ROC_gt"
        )
        self.log(f"{self._mode}_auroc_gt", auroc_gt, on_epoch=True, prog_bar=False)

        if self._exp["visu"]["log_test_video"]:
            inp = os.path.join(self._visualizer._p_visu, f"{self._mode}/graph_trav")
            out = os.path.join(self._visualizer._p_visu, f"{self._mode}/graph_trav.mp4")
            cmd = f"ffmpeg -framerate 2 -pattern_type glob -i '{inp}/*_test_GraphTrav.png' -c:v libx264 -pix_fmt yuv420p {out}"
            os.system(cmd)

            inp = os.path.join(self._visualizer._p_visu, f"{self._mode}/graph_conf")
            out2 = os.path.join(self._visualizer._p_visu, f"{self._mode}/graph_conf.mp4")
            cmd = f"ffmpeg -framerate 2 -pattern_type glob -i '{inp}/*_test_GraphConf.png' -c:v libx264 -pix_fmt yuv420p {out2}"
            os.system(cmd)

            try:
                self.logger.experiment["graph_trav"].upload(out)
                self.logger.experiment["graph_conf"].upload(out)
            except:
                pass

        detailed_test_results = {
            "test_roc_gt_fpr": fpr_gt.cpu().numpy(),
            "test_roc_gt_tpr": tpr_gt.cpu().numpy(),
            "test_roc_gt_thresholds": thresholds_gt.cpu().numpy(),
            "test_auroc_gt": auroc_gt,
            "test_roc_prop_fpr": fpr_pro.cpu().numpy(),
            "test_roc_prop_tpr": tpr_pro.cpu().numpy(),
            "test_roc_prop_thresholds": thresholds_pro.cpu().numpy(),
            "test_auroc_prop": auroc_pro,
        }

        with open(os.path.join(self._exp["general"]["model_path"], "detailed_test_results.pkl"), "wb") as handle:
            pickle.dump(detailed_test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return detailed_test_results

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
