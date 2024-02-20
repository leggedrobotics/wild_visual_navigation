#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.utils import ConfidenceGenerator

import torch.nn.functional as F
from wild_visual_navigation.utils import Data

import torch
from typing import Optional
from torch import nn


class AnomalyLoss(nn.Module):
    def __init__(
        self,
        confidence_std_factor: float,
        method: str,
        log_enabled: bool,
        log_folder: str,
    ):
        super(AnomalyLoss, self).__init__()

        self._confidence_generator = ConfidenceGenerator(
            std_factor=confidence_std_factor, method=method, log_enabled=log_enabled, log_folder=log_folder
        )

    def forward(
        self,
        graph: Optional[Data],
        res: dict,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
    ):
        loss_aux = {}
        loss_aux["loss_trav"] = torch.tensor([0.0])
        loss_aux["loss_reco"] = torch.tensor([0.0])

        losses = res["logprob"].sum(1) + res["log_det"]  # Sum over all channels, resulting in h*w output dimensions

        if update_generator:
            confidence = self._confidence_generator.update(
                x=-losses.clone().detach(), x_positive=-losses.clone().detach(), step=step
            )

        loss_aux["confidence"] = confidence

        return -torch.mean(losses), loss_aux, confidence

    def update_node_confidence(self, node):
        node.confidence = 0


class TraversabilityLoss(nn.Module):
    def __init__(
        self,
        w_trav: float,
        w_reco: float,
        w_temp: float,
        anomaly_balanced: bool,
        model: nn.Module,
        method: str,
        confidence_std_factor: float,
        log_enabled: bool,
        log_folder: str,
        trav_cross_entropy=False,
    ):
        # TODO remove trav_cross_entropy default param when running in online mode
        super(TraversabilityLoss, self).__init__()
        self._w_trav = w_trav

        self._w_reco = w_reco
        self._w_temp = w_temp
        self._model = model
        self._anomaly_balanced = anomaly_balanced
        self._trav_cross_entropy = trav_cross_entropy
        if self._trav_cross_entropy:
            self._trav_loss_func = F.binary_cross_entropy
        else:
            self._trav_loss_func = F.mse_loss

        self._confidence_generator = ConfidenceGenerator(
            std_factor=confidence_std_factor, method=method, log_enabled=log_enabled, log_folder=log_folder
        )

    def reset(self):
        if self._anomaly_balanced:
            self._confidence_generator.reset()

    def forward(
        self,
        graph: Data,
        res: torch.Tensor,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
    ):
        # Compute reconstruction loss
        nr_channel_reco = graph.x.shape[1]
        loss_reco = F.mse_loss(res[:, -nr_channel_reco:], graph.x, reduction="none").mean(dim=1)

        with torch.no_grad():
            if update_generator:
                confidence = self._confidence_generator.update(
                    x=loss_reco,
                    x_positive=loss_reco[graph.y_valid],
                    step=step,
                    log_step=log_step,
                )
            else:
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)

        label = graph.y[:]
        if self._trav_cross_entropy:
            label = label.type(torch.long)
            loss_trav_raw = self._trav_loss_func(
                res[:, :-nr_channel_reco].squeeze()[:, 0],
                label.type(torch.float32),
                reduction="none",
            )
        else:
            loss_trav_raw = self._trav_loss_func(res[:, :-nr_channel_reco].squeeze(), label, reduction="none")

        ele = graph.y_valid.shape[0]  # 400 #
        selector = torch.zeros_like(graph.y_valid)
        selector[:ele] = 1
        loss_trav_raw_labeled = loss_trav_raw[graph.y_valid * selector]
        loss_trav_raw_not_labeled = loss_trav_raw[~graph.y_valid * selector]

        # Scale the loss
        loss_trav_raw_not_labeled_weighted = loss_trav_raw_not_labeled * (1 - confidence)[~graph.y_valid * selector]

        if self._anomaly_balanced:
            loss_trav_confidence = (loss_trav_raw_not_labeled_weighted.sum() + loss_trav_raw_labeled.sum()) / (
                graph.y.shape[0]
            )
        else:
            loss_trav_confidence = loss_trav_raw[selector].mean()

        loss_temp = torch.zeros_like(loss_trav_confidence)

        loss_reco_mean = loss_reco[graph.y_valid * selector].mean()
        # Compute total loss
        loss = self._w_trav * loss_trav_confidence + self._w_reco * loss_reco_mean + self._w_temp * loss_temp

        res_updated = res
        return (
            loss,
            {
                "loss_reco": loss_reco_mean,
                "loss_trav": loss_trav_raw.mean(),
                "loss_temp": loss_temp.mean(),
                "loss_trav_confidence": loss_trav_confidence,
                "confidence": confidence,
            },
            res_updated,
        )

    def update_node_confidence(self, node):
        reco_loss = F.mse_loss(node.prediction[:, 1:], node.features, reduction="none").mean(dim=1)
        node.confidence = self._confidence_generator.inference_without_update(reco_loss)
