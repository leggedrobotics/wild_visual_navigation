#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
# author: Jonas Frey

import torch


class SegmentExtractor(torch.nn.Module):
    @torch.no_grad()
    def __init__(self):
        super().__init__()
        # Use Convolutional Filter to Extract Edges
        self.f1 = torch.nn.Conv2d(1, 4, (3, 3), padding_mode="replicate", padding=(3, 3), bias=False)
        self.f1.weight[:, :, :, :] = 0
        # 0  0  0
        # 0  1 -1
        # 0  0  0
        self.f1.weight[0, 0, 1, 1] = 1
        self.f1.weight[0, 0, 1, 2] = -1
        # 0  0  0
        # 1 -1  0
        # 0  0  0
        self.f1.weight[1, 0, 1, 0] = 1
        self.f1.weight[1, 0, 1, 1] = -1
        # 0  0  0
        # 0  1  0
        # 0 -1  0
        self.f1.weight[2, 0, 1, 1] = 1
        self.f1.weight[2, 0, 2, 1] = -1
        # 0  0  0
        # 0 -1  0
        # 0  1  0
        self.f1.weight[3, 0, 0, 1] = 1
        self.f1.weight[3, 0, 1, 1] = -1

    @torch.no_grad()
    def adjacency_list(self, seg: torch.tensor):
        """Extracts a adjacency list based on neigboring classes.

        Args:
            seg (torch.Tensor, dtype=torch.long, shape=(BS, 1, H, W)): Segmentation

        Returns:
            adjacency_list (torch.Tensor, dtype=torch.long, shape=(N, 2): Adjacency list of undirected graph
        """
        assert seg.shape[0] == 1 and len(seg.shape) == 4, f"{seg.shape}"

        res = self.f1(seg.type(torch.float32))
        boundary_mask = (res != 0)[0, :, 2:-2, 2:-2]

        # Shifting the filter allows to index the left and right segment of a bordered
        left_idx = torch.cat([seg[0, 0, boundary_mask[0]], seg[0, 0, boundary_mask[2]]])
        right_idx = torch.cat([seg[0, 0, boundary_mask[1]], seg[0, 0, boundary_mask[3]]])

        # Create adjacency_list based on the given pairs (crucial to use float64 here)
        div = seg.max() + 1
        unique_key = (left_idx + (right_idx * (div))).type(torch.float64)
        m = torch.unique(unique_key)

        le_idx = (m % div).type(torch.long)
        ri_idx = torch.floor(m / div).type(torch.long)
        adjacency_list = torch.stack([le_idx, ri_idx], dim=1)

        return adjacency_list

    @torch.no_grad()
    def centers(self, seg: torch.tensor):
        """Extracts a center position in image plane of clusters.

        Args:
            seg (torch.Tensor, dtype=torch.long, shape=(BS, 1, H, W)): Segmentation

        Returns:
            centers (torch.Tensor, dtype=torch.long, shape=(N, 2): Center position in image plane of clusters.
        """

        assert seg.shape[0] == 1 and len(seg.shape) == 4

        centers = []
        tmp_seg = seg.permute(
            *torch.arange(seg.ndim - 1, -1, -1)
        )  # complicated command because seg.T will be deprecated
        for s in range(seg.max() + 1):
            indices = torch.nonzero((s == tmp_seg)[:, :, 0, 0])
            res = indices.type(torch.float32).mean(dim=0)
            centers.append(res)
        centers = torch.stack(centers)

        return centers
