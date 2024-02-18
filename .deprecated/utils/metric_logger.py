import torch
from torchmetrics import ROC, AUROC, Accuracy


class MetricLogger(torch.nn.Module):
    def __init__(self, log_handel, log_roc_image):
        super().__init__()
        self.log_roc_image = log_roc_image
        modes = ["train_metric", "test_metric", "val_metric"]
        # SEGMENT SPACE
        self.roc_gt_seg = torch.nn.ModuleDict({m: ROC(task="binary", thresholds=5000) for m in modes})
        self.roc_self_seg = torch.nn.ModuleDict({m: ROC(task="binary", thresholds=5000) for m in modes})
        self.auroc_gt_seg = torch.nn.ModuleDict({m: AUROC(task="binary") for m in modes})
        self.auroc_self_seg = torch.nn.ModuleDict({m: AUROC(task="binary") for m in modes})

        # IMAGE SPACE
        if self.log_roc_image:
            # ROC
            self.roc_gt_image = torch.nn.ModuleDict({m: ROC(task="binary", thresholds=5000) for m in modes})
            self.roc_self_image = torch.nn.ModuleDict({m: ROC(task="binary", thresholds=5000) for m in modes})
        # AUROC
        self.auroc_gt_image = torch.nn.ModuleDict({m: AUROC(task="binary") for m in modes})
        self.auroc_self_image = torch.nn.ModuleDict({m: AUROC(task="binary") for m in modes})
        # AUROC ANOMALY
        self.auroc_anomaly_gt_image = torch.nn.ModuleDict({m: AUROC(task="binary") for m in modes})
        self.auroc_anomaly_self_image = torch.nn.ModuleDict({m: AUROC(task="binary") for m in modes})
        # ACC
        self.acc_gt_image = torch.nn.ModuleDict({m: Accuracy(task="binary") for m in modes})
        self.acc_self_image = torch.nn.ModuleDict({m: Accuracy(task="binary") for m in modes})
        # ACC ANOMALY
        self.acc_anomaly_gt_image = torch.nn.ModuleDict({m: Accuracy(task="binary") for m in modes})
        self.acc_anomaly_self_image = torch.nn.ModuleDict({m: Accuracy(task="binary") for m in modes})
        self.log_handel = log_handel

    @torch.no_grad()
    def log_segment(self, graph, res, mode):
        mode = mode + "_metric"
        # ROC
        self.roc_gt_seg[mode](preds=res, target=graph.label.type(torch.long))
        self.roc_self_seg[mode](preds=res, target=graph.y.type(torch.long))
        # AUROC
        self.auroc_gt_seg[mode](preds=res, target=graph.label.type(torch.long))
        self.auroc_self_seg[mode](preds=res, target=graph.y.type(torch.long))
        self.log_handel(
            f"{mode}_auroc_gt_seg",
            self.auroc_gt_seg[mode],
            on_epoch=True,
            on_step=False,
            batch_size=graph.label.shape[0],
        )
        self.log_handel(
            f"{mode}_auroc_self_seg",
            self.auroc_self_seg[mode],
            on_epoch=True,
            on_step=False,
            batch_size=graph.label.shape[0],
        )

    @torch.no_grad()
    def log_image(self, graph, res, mode, threshold=0.5):
        mode = mode + "_metric"
        # project graph predictions and label onto the image
        bp = graph.label.clone().type(torch.float32).flatten()
        bpro = graph.label.clone().type(torch.float32).flatten()

        BS, H, W = graph.label.shape
        # Use the position within the batch to offset the segmentation
        # This avoids iterating over the batch dimension
        batch_pixel_index_offset = graph.ptr[:-1, None, None].repeat(1, H, W)
        # B, H, W
        seg_pixel_index = (graph.seg + batch_pixel_index_offset).flatten()

        bp = res[seg_pixel_index, 0].reshape(BS, H, W)
        bpro = graph.y[seg_pixel_index].reshape(BS, H, W)

        if self.log_roc_image:
            # ROC
            self.roc_gt_image[mode](preds=bp, target=graph.label.type(torch.long))
            self.roc_self_image[mode](preds=bp, target=bpro.type(torch.long))

        # AUROC
        self.auroc_gt_image[mode](preds=bp, target=graph.label.type(torch.long))
        self.auroc_self_image[mode](preds=bp, target=bpro.type(torch.long))

        self.log_handel(
            f"{mode}_auroc_gt_image",
            self.auroc_gt_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )
        self.log_handel(
            f"{mode}_auroc_self_image",
            self.auroc_self_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )

        # GT
        self.acc_gt_image[mode](preds=bp > threshold, target=graph.label.type(torch.long))
        self.acc_self_image[mode](preds=bp > threshold, target=bpro.type(torch.long))

        self.log_handel(
            f"{mode}_acc_gt_image",
            self.acc_gt_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )
        self.log_handel(
            f"{mode}_acc_self_image",
            self.acc_self_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )

    @torch.no_grad()
    def log_confidence(self, graph, res, confidence, mode):
        mode = mode + "_metric"
        BS, H, W = graph.label.shape
        batch_pixel_index_offset = graph.ptr[:-1, None, None].repeat(1, H, W)
        # B, H, W
        seg_pixel_index = (graph.seg + batch_pixel_index_offset).flatten()

        bp = graph.label.clone().type(torch.float32).flatten()
        bpro = graph.label.clone().type(torch.float32).flatten()
        buffer_conf = graph.label.clone().type(torch.float32).flatten()

        bp = res[seg_pixel_index, 0].reshape(BS, H, W)  # noqa: F841
        bpro = graph.y[seg_pixel_index].reshape(BS, H, W)
        buffer_conf = confidence[seg_pixel_index].reshape(BS, H, W)

        self.auroc_anomaly_gt_image[mode](preds=buffer_conf, target=bpro.type(torch.long))
        self.auroc_anomaly_self_image[mode](preds=buffer_conf, target=graph.label.type(torch.long))
        self.log_handel(
            f"{mode}_auroc_anomaly_gt_image",
            self.auroc_anomaly_gt_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )
        self.log_handel(
            f"{mode}_auroc_anomaly_self_image",
            self.auroc_anomaly_self_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )

        self.acc_anomaly_gt_image[mode](preds=buffer_conf, target=graph.label.type(torch.long))
        self.acc_anomaly_self_image[mode](preds=buffer_conf, target=bpro.type(torch.long))
        self.log_handel(
            f"{mode}_acc_anomaly_gt_image",
            self.acc_anomaly_gt_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )
        self.log_handel(
            f"{mode}_acc_anomaly_self_image",
            self.acc_anomaly_self_image[mode],
            on_epoch=True,
            on_step=False,
            batch_size=BS,
        )

    def get_epoch_results(self, mode):
        mo = mode
        mode = mode + "_metric"
        # label is the gt label
        if self.log_roc_image:
            roc_gt_image = [a.cpu().numpy() for a in self.roc_gt_image[mode].compute()]
        else:
            roc_gt_image = [None, None, None]
        auroc_gt_image = self.auroc_gt_image[mode].compute().item()

        # generate supervision label
        if self.log_roc_image:
            roc_self_image = [a.cpu().numpy() for a in self.roc_self_image[mode].compute()]
        else:
            roc_self_image = [None, None, None]
        auroc_self_image = self.auroc_self_image[mode].compute().item()

        res = {
            f"{mo}_roc_gt_image": roc_gt_image,
            f"{mo}_roc_self_image": roc_self_image,
            f"{mo}_auroc_gt_image": auroc_gt_image,
            f"{mo}_auroc_self_image": auroc_self_image,
            f"{mo}_auroc_anomaly_gt_image": self.auroc_anomaly_gt_image[mode].compute().item(),
            f"{mo}_auroc_anomaly_self_image": self.auroc_anomaly_self_image[mode].compute().item(),
            f"{mo}_acc_gt_image": self.acc_gt_image[mode].compute().item(),
            f"{mo}_acc_self_image": self.acc_self_image[mode].compute().item(),
            f"{mo}_acc_anomaly_gt_image": self.acc_anomaly_gt_image[mode].compute().item(),
            f"{mo}_acc_anomaly_self_image": self.acc_anomaly_self_image[mode].compute().item(),
        }
        return res

    def reset(self, mode):
        mode = mode + "_metric"

        self.roc_gt_seg[mode].reset()
        self.roc_self_seg[mode].reset()
        self.auroc_gt_seg[mode].reset()
        self.auroc_self_seg[mode].reset()

        # IMAGE SPACE
        # ROC
        if self.log_roc_image:
            self.roc_gt_image[mode].reset()
            self.roc_self_image[mode].reset()
        # AUROC
        self.auroc_gt_image[mode].reset()
        self.auroc_self_image[mode].reset()
        # AUROC ANOMALY
        self.auroc_anomaly_gt_image[mode].reset()
        self.auroc_anomaly_self_image[mode].reset()
        # ACC
        self.acc_gt_image[mode].reset()
        self.acc_self_image[mode].reset()
        # ACC ANOMALY
        self.acc_anomaly_gt_image[mode].reset()
        self.acc_anomaly_self_image[mode].reset()
