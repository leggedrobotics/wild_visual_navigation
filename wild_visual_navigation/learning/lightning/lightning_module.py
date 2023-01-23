from wild_visual_navigation.learning.model import get_model
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
        self._visualizer = LearningVisualizer(**exp["visu"]["learning_visu"], pl_model=self)
        self._exp = exp
        self._env = env
        self._mode = "train"
        self._log = log

        self._validation_roc_gt_image = ROC(task="binary")
        self._validation_auroc_gt_image = AUROC(task="binary")
        self._validation_roc_proprioceptive_image = ROC(task="binary")
        self._validation_auroc_proprioceptive_image = AUROC(task="binary")
        self._validation_roc_proprioceptive = ROC(task="binary")
        self._validation_auroc_proprioceptive = AUROC(task="binary")

        # The Accumulated results are used to store the results of the validation dataloader for every validation epoch. Allows after training to have summary avaiable.
        self.accumulated_val_results = []

        self._test_roc_proprioceptive_image = ROC(task="binary")
        self._test_roc_gt_image = ROC(task="binary")
        self._test_auroc_proprioceptive_image = AUROC(task="binary")
        self._test_auroc_gt_image = AUROC(task="binary")
        self._test_auroc_anomaly_proprioceptive_image = AUROC(task="binary")
        self._test_auroc_anomaly_gt_image = AUROC(task="binary")

        # The Accumulated results are used to store the results of the test dataloader. Allows after testing to have summary avaiable.
        self.accumulated_test_results = []

        # This flag is currently set in a hack way before straining the testing.
        # It is only used to log the test results to the disk.
        # TODO remove this part.
        self.nr_test_run = -1
        self._val_step = 0

        self._traversability_loss = TraversabilityLoss(**self._exp["loss"], model=self._model)

    def forward(self, data: torch.tensor):
        return self._model(data)

    # TRAINING
    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visualizer.epoch = self.current_epoch

    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        graph = batch
        if self.global_step > 200:
            self._traversability_loss.schedule_w_trav()

        nr = self._exp["general"]["store_model_every_n_steps"]
        if type(nr) == int:
            if self.global_step % nr == 0:
                path = os.path.join(
                    self._exp["general"]["model_path"],
                    self._exp["general"]["store_model_every_n_steps_key"] + f"_{self.global_step}.pt",
                )
                torch.save(self.state_dict(), path)

        BS = graph.ptr.numel() - 1

        res = self._model(graph)
        loss, loss_aux = self._traversability_loss(graph, res)

        for k, v in loss_aux.items():
            self.log(f"{self._mode}_{k}", v.item(), on_epoch=True, prog_bar=True, batch_size=BS)
        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.visu(graph, res)
        return loss

    def visu(self, graph: Data, res: torch.tensor):
        try:
            self.log(
                "confidence_mean",
                self._traversability_loss._confidence_generator.mean.item(),
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                "confidence_std",
                self._traversability_loss._confidence_generator.std.item(),
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        except:
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
                    pred[:, 0], seg, graph[b], center, img, not_log=True, colorize_invalid_centers=True
                )
                t2 = self._visualizer.plot_traversability_graph_on_seg(
                    graph[b].y, seg, graph[b], center, img, not_log=True
                )
                t_img = self._visualizer.plot_image(img, not_log=True)
                self._visualizer.plot_list(
                    imgs=[t1, t2, t_img], tag=f"C{c}_{self._mode}_GraphTrav", store_folder=f"{self._mode}/graph_trav"
                )

                reco_loss = F.mse_loss(pred[:, 1:], graph[b].x, reduction="none").mean(dim=1)
                conf = self._traversability_loss._confidence_generator.inference_without_update(reco_loss)

                # # Visualize Graph with Segmentation
                c1 = self._visualizer.plot_traversability_graph_on_seg(
                    conf, seg, graph[b], center, img, not_log=True, colorize_invalid_centers=True
                )
                c2 = self._visualizer.plot_traversability_graph_on_seg(
                    graph[b].y_valid.type(torch.float32), seg, graph[b], center, img, not_log=True
                )
                c_img = self._visualizer.plot_image(img, not_log=True)
                self._visualizer.plot_list(
                    imgs=[c1, c2, c_img], tag=f"C{c}_{self._mode}_GraphConf", store_folder=f"{self._mode}/graph_conf"
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

            # Logging the confidence
            mean = self._traversability_loss._confidence_generator.mean.item()
            std = self._traversability_loss._confidence_generator.std.item()
            reco_loss = F.mse_loss(pred[:, 1:], graph[b].x, reduction="none").mean(dim=1)
            confidence = self._traversability_loss._confidence_generator.inference_without_update(reco_loss)

            self._visualizer.plot_histogram(
                reco_loss, graph[b].y, mean, std, tag=f"C{c}_{self._mode}__confidence_generator_prop"
            )

            if hasattr(graph[b], "y_gt"):
                self._visualizer.plot_histogram(
                    reco_loss, graph[b].y_gt, mean, std, tag=f"C{c}_{self._mode}__confidence_generator_gt"
                )

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        # Log epoch metric
        self._mode = "train"

    # VALIDATION
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0
        self._validation_roc_gt_image.reset()
        self._validation_auroc_gt_image.reset()
        self._validation_roc_proprioceptive_image.reset()
        self._validation_auroc_proprioceptive_image.reset()
        self._validation_roc_proprioceptive.reset()
        self._validation_auroc_proprioceptive.reset()
        self._val_step = 0

    def log_metrics(
        self,
        roc_gt_image,
        auroc_gt_image,
        roc_proprioceptive_image,
        auroc_proprioceptive_image,
        graph,
        res,
        debug=False,
    ):
        # project graph predictions and label onto the image
        buffer_pred = graph.label.clone().type(torch.float32).flatten()
        buffer_prop = graph.label.clone().type(torch.float32).flatten()

        BS, H, W = graph.label.shape
        # Use the position within the batch to offset the segmentation
        # This avoids iterating over the batch dimension
        batch_pixel_index_offset = graph.ptr[:-1, None, None].repeat(1, H, W)
        # B, H, W
        seg_pixel_index = (graph.seg + batch_pixel_index_offset).flatten()

        buffer_pred = res[seg_pixel_index, 0].reshape(BS, H, W)
        buffer_prop = graph.y[seg_pixel_index].reshape(BS, H, W)

        # label is the gt label
        roc_gt_image.update(preds=buffer_pred, target=graph.label)
        auroc_gt_image.update(preds=buffer_pred, target=graph.label)

        # generate proprioceptive label
        roc_proprioceptive_image.update(preds=buffer_pred, target=buffer_prop)
        auroc_proprioceptive_image.update(preds=buffer_pred, target=buffer_prop)

        if self._mode == "test" and self._exp["loss"]["w_reco"] != 0:
            reco_loss = F.mse_loss(res[:, 1:], graph.x, reduction="none").mean(dim=1)
            conf = self._traversability_loss._confidence_generator.inference_without_update(reco_loss)
            buffer_conf = graph.label.clone().type(torch.float32).flatten()
            buffer_conf = conf[seg_pixel_index].reshape(BS, H, W)

            self._test_auroc_anomaly_proprioceptive_image.update(preds=buffer_conf, target=buffer_prop)
            self._test_auroc_anomaly_gt_image.update(preds=buffer_conf, target=graph.label)

        if debug:
            b = 0
            # Visualize the labels quickly
            i1 = self._visualizer.plot_detectron(
                graph.img[b],
                (buffer_pred[b] * 255).type(torch.long),
                not_log=True,
                colorize_invalid_centers=True,
                max_seg=256,
                colormap="RdYlBu",
            )
            i2 = self._visualizer.plot_detectron(
                graph.img[b],
                (buffer_prop[b] * 255).type(torch.long),
                not_log=True,
                colorize_invalid_centers=True,
                max_seg=256,
                colormap="RdYlBu",
            )
            i3 = self._visualizer.plot_detectron(
                graph.img[b],
                (graph.label[b] * 255).type(torch.long),
                not_log=True,
                colorize_invalid_centers=True,
                max_seg=256,
                colormap="RdYlBu",
            )
            img = self._visualizer.plot_list(imgs=[i1, i2, i3], tag=f"Pred_Prop_GT", store_folder=f"{self._mode}/debug")
            from PIL import Image

            img = Image.fromarray(img)
            img.show()

    def validation_step(self, batch: any, batch_idx: int, dataloader_id: int = 0) -> torch.Tensor:
        graph = batch
        BS = graph.ptr.numel() - 1

        res = self._model(graph)

        loss, loss_aux = self._traversability_loss(graph, res)

        self._validation_roc_proprioceptive.update(preds=res[:, 0], target=(graph.y > 0).type(torch.long))
        self._validation_auroc_proprioceptive.update(res[:, 0], (graph.y > 0).type(torch.long))

        if hasattr(graph, "label"):
            self.log_metrics(
                self._validation_roc_gt_image,
                self._validation_auroc_gt_image,
                self._validation_roc_proprioceptive_image,
                self._validation_auroc_proprioceptive_image,
                graph,
                res,
            )

        for k, v in loss_aux.items():
            self.log(f"{self._mode}_{k}", v.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.visu(graph, res)
        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        fpr, tpr, thresholds = self._validation_roc_proprioceptive.compute()
        auroc = self._validation_auroc_proprioceptive.compute().item()

        if False:
            self._visualizer.plot_roc(
                x=fpr.cpu().numpy(), y=tpr.cpu().numpy(), y_tag=f"AUCROC_{auroc:.4f}", tag=f"ROC_{self._mode}"
            )

        self.log(f"{self._mode}_auroc", auroc, on_epoch=True, prog_bar=False)

        if self._exp["visu"]["log_val_video"]:
            inp = os.path.join(self._visualizer._p_visu, f"{self._mode}/graph_trav")
            out2 = os.path.join(self._visualizer._p_visu, f"{self._mode}/epoch_{self.current_epoch}_graph_trav.mp4")
            cmd = f"ffmpeg -framerate 2 -pattern_type glob -i '{inp}/{self.current_epoch}*_{self._mode}_GraphTrav.png' -c:v libx264 -pix_fmt yuv420p {out2}"
            os.system(cmd)

        if self._exp["ablation_data_module"]["val_equals_test"]:
            # label is the gt label
            validation_roc_gt_image = self._validation_roc_gt_image.compute()
            validation_roc_gt_image = [a.cpu().numpy() for a in validation_roc_gt_image]
            validation_auroc_gt_image = self._validation_auroc_gt_image.compute().item()

            # generate proprioceptive label
            validation_roc_proprioceptive_image = self._validation_roc_proprioceptive_image.compute()
            validation_roc_proprioceptive_image = [a.cpu().numpy() for a in validation_roc_proprioceptive_image]
            validation_auroc_proprioceptive_image = self._validation_auroc_proprioceptive_image.compute().item()
            res = {
                "validation_roc_gt_image": validation_roc_gt_image,
                "validation_auroc_gt_image": validation_auroc_gt_image,
                "validation_roc_proprioceptive_image": validation_roc_proprioceptive_image,
                "validation_auroc_proprioceptive_image": validation_auroc_proprioceptive_image,
            }
            self.accumulated_val_results.append(res)

            for k, v in res.items():
                if k.find("auroc") != -1:
                    self.log(k, v, on_epoch=True)

    # TESTING
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0

    def test_step(self, batch: any, batch_idx: int, dataloader_id: int = 0) -> torch.Tensor:
        graph = batch
        BS = graph.ptr.numel() - 1

        res = self._model(graph)

        loss, loss_aux = self._traversability_loss(graph, res)

        # project graph predictions and label onto the image
        self.log_metrics(
            self._test_roc_gt_image,
            self._test_auroc_gt_image,
            self._test_roc_proprioceptive_image,
            self._test_auroc_proprioceptive_image,
            graph,
            res,
        )

        for k, v in loss_aux.items():
            self.log(f"{self._mode}_{k}", v.item(), on_epoch=True, prog_bar=True, batch_size=BS)
        self.log(f"{self._mode}_loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=BS)

        self.visu(graph, res)

        return loss

    def test_epoch_end(self, outputs: any, plot=False):
        ################ NEW VERSION ################
        # label is the gt label
        test_roc_gt_image = self._test_roc_gt_image.compute()
        test_roc_gt_image = [a.cpu().numpy() for a in test_roc_gt_image]
        test_auroc_gt_image = self._test_auroc_gt_image.compute().item()

        # generate proprioceptive label
        test_roc_proprioceptive_image = self._test_roc_proprioceptive_image.compute()
        test_roc_proprioceptive_image = [a.cpu().numpy() for a in test_roc_proprioceptive_image]
        test_auroc_proprioceptive_image = self._test_auroc_proprioceptive_image.compute().item()

        dic = {
            "test_roc_gt_image": test_roc_gt_image,
            "test_auroc_gt_image": test_auroc_gt_image,
            "test_roc_proprioceptive_image": test_roc_proprioceptive_image,
            "test_auroc_proprioceptive_image": test_auroc_proprioceptive_image,
        }

        if self._exp["loss"]["w_reco"] != 0:
            test_auroc_anomaly_proprioceptive_image = self._test_auroc_anomaly_proprioceptive_image.compute().item()
            test_auroc_anomaly_gt_image = self._test_auroc_anomaly_gt_image.compute().item()
            dic["test_auroc_anomaly_proprioceptive_image"] = test_auroc_anomaly_proprioceptive_image
            dic["test_auroc_anomaly_gt_image"] = test_auroc_anomaly_gt_image

        dic2 = {k: v.item() for k, v in self.trainer.logged_metrics.items() if k.find("step") == -1}
        dic.update(dic2)

        self.accumulated_test_results.append(dic)
        ################ NEW VERSION FINISHED  ################

        # potentially broken or deprecated code
        dtr = {}
        fpr_pro, tpr_pro, thresholds_pro = test_roc_proprioceptive_image
        auroc_pro = test_auroc_proprioceptive_image
        fpr_gt, tpr_gt, thresholds_gt = test_roc_gt_image
        auroc_gt = test_auroc_gt_image
        if plot:
            self._visualizer.plot_roc(
                x=fpr_pro,
                y=tpr_pro,
                y_tag=f"AUCROC_{auroc_pro:.4f}",
                tag=f"{self._mode}_ROC_proprioceptive_{self.nr_test_run}",
            )
            self.log(
                f"{self._mode}_auroc_proprioceptive_{self.nr_test_run}",
                auroc_pro,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                batch_size=1,
            )

            self._visualizer.plot_roc(
                x=fpr_gt,
                y=tpr_gt,
                y_tag=f"AUCROC_{auroc_gt:.4f}_{self.nr_test_run}",
                tag=f"{self._mode}_ROC_gt_{self.nr_test_run}",
            )
            self.log(
                f"{self._mode}_auroc_gt_{self.nr_test_run}",
                auroc_gt,
                on_epoch=True,
                prog_bar=False,
                on_step=False,
                batch_size=1,
            )

        dtr[f"test_roc_gt_fpr"] = (fpr_gt,)
        dtr[f"test_roc_gt_tpr"] = (tpr_gt,)
        dtr[f"test_roc_gt_thresholds"] = (thresholds_gt,)
        dtr[f"test_auroc_gt"] = auroc_gt
        dtr[f"test_roc_prop_fpr"] = fpr_pro
        dtr[f"test_roc_prop_tpr"] = tpr_pro
        dtr[f"test_roc_prop_thresholds"] = thresholds_pro
        dtr[f"test_auroc_prop"] = auroc_pro

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

        if self._exp["general"]["log_to_disk"]:
            with open(
                os.path.join(self._exp["general"]["model_path"], f"{self.nr_test_run}_detailed_test_results.pkl"), "wb"
            ) as handle:
                pickle.dump(dtr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._test_roc_proprioceptive_image.reset()
        self._test_roc_gt_image.reset()
        self._test_auroc_proprioceptive_image.reset()
        self._test_auroc_gt_image.reset()
        self._test_auroc_anomaly_proprioceptive_image.reset()
        self._test_auroc_anomaly_gt_image.reset()

        return dtr

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self._exp["optimizer"]["name"] == "ADAM":
            return torch.optim.Adam(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
        elif self._exp["optimizer"]["name"] == "SGD":
            return torch.optim.SGD(self._model.parameters(), lr=self._exp["optimizer"]["lr"])
