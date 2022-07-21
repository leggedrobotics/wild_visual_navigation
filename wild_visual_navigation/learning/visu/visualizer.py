# TODO: Jonas doc strings, rework visualiation functions

import os
import numpy as np
import imageio
import cv2
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import cm
from wild_visual_navigation.utils import get_img_from_fig
from wild_visual_navigation.learning.visu import image_functionality
import torch

__all__ = ["LearningVisualizer"]


class LearningVisualizer:
    def __init__(self, p_visu, store, pl_model, epoch=0, num_classes=22):
        self._p_visu = p_visu
        self._pl_model = pl_model
        self._epoch = epoch
        self._store = store

        if not os.path.exists(self._p_visu):
            os.makedirs(self._p_visu)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, store):
        self._store = store

    @image_functionality
    def plot_graph_result(self, graph, center, img, seg, res, use_seg=False, **kwargs):
        """Plot Graph GT and Predtion on Image

        Args:
            graph (torch_geometric.data.data.Data): Graph
            center (torch.Tensor,  dtype=torch.float32, shape=(1, N, 2)): Center positions in image plane
            img (torch.Tensor,  dtype=torch.float32, shape=(3, H, W)): Image
            seg (torch.Tensor,  dtype=torch.int64, shape=(H, W)): Segmentation mask
            res (torch.Tensor,  dtype=torch.float32, shape=(N)): Perdicted label
            use_seg (bool, optional): Plot the segmentation mask. Defaults to False.

        Returns:
            img (np.Array,  dtype=np.uint8, shape=(3, H, W)): Left prediction, right ground truth
        """
        max_seg = 1.0
        elipse_size = 5

        y = (graph.y.type(torch.float32) / max_seg * 255).type(torch.uint8)
        y_pred = (res.clip(0, max_seg) / max_seg * 255).type(torch.uint8)
        img = np.uint8((img.permute(1, 2, 0) * 255).cpu().numpy())

        img_pil = Image.fromarray(img)
        img_pred_pil = Image.fromarray(np.copy(img))
        img_draw = ImageDraw.Draw(img_pil)
        img_pred_draw = ImageDraw.Draw(img_pred_pil)

        if use_seg:
            seg = self.plot_segmentation(seg, max_seg=100, not_log=True)
            seg_pil = Image.fromarray(seg)
            seg_draw = ImageDraw.Draw(seg_pil)

        adjacency_list = graph.edge_index

        colormap = "viridis"
        viridis = cm.get_cmap(colormap, 255)
        col_map = np.uint8(viridis(np.linspace(0, 1, 256))[:, :3] * 255)


        for i in range(adjacency_list.shape[1]):
            a, b = adjacency_list[0, i], adjacency_list[1, i]
            line_params = center[a].tolist() + center[b].tolist()
            img_draw.line(line_params, fill=(255, 0, 0))
            img_pred_draw.line(line_params, fill=(255, 0, 0))
            if use_seg:
                seg_draw.line(line_params, fill=(255, 0, 0))

        for i in range(center.shape[0]):
            params = center[i].tolist()
            params = [p - elipse_size for p in params] + [p + elipse_size for p in params]
            img_draw.ellipse(params, width=10, fill=tuple(col_map[y[i]].tolist()))
            img_pred_draw.ellipse(params, width=10, fill=tuple(col_map[y_pred[i]].tolist()))
            if use_seg:
                seg_draw.ellipse(params, width=10, fill=tuple(col_map[y[i]].tolist()))

        img_res = np.array(img_pil)
        img_pred_res = np.array(img_pred_pil)

        ls = [img_pred_res, img_res]
        if use_seg:
            ls = ls + [np.array(seg_pil)]
        res = np.concatenate(ls, axis=1)
        return res

    @image_functionality
    def plot_segmentation(self, seg, max_seg=40, colormap="viridis", **kwargs):
        try:
            seg = seg.clone().cpu().numpy()
        except:
            pass

        if seg.dtype == np.bool:
            max_seg = 2

        viridis = cm.get_cmap(colormap, max_seg)
        col_map = np.uint8(viridis(np.linspace(0, 1, max_seg))[:, :3] * 255)

        H, W = seg.shape[:2]
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for i, color in enumerate(col_map):
            img[seg == i] = color[:3]

        return img

    @image_functionality
    def plot_image(self, img, **kwargs):
        """
        ----------
        img : CHW HWC accepts torch.tensor or numpy.array
            Range 0-1 or 0-255
        """
        try:
            img = img.clone().cpu().numpy()
        except:
            pass

        if img.shape[2] == 3:
            pass
        elif img.shape[0] == 3:
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        else:
            raise Exception("Invalid Shape")
        if img.max() <= 1:
            img = img * 255

        img = np.uint8(img)
        return img

    @image_functionality
    def plot_detectron(
        self,
        img,
        label,
        text_off=False,
        alpha=0.5,
        draw_bound=True,
        shift=2.5,
        font_size=12,
        **kwargs,
    ):
        """
        ----------
        img : CHW HWC accepts torch.tensor or numpy.array
            Range 0-1 or 0-255
        label: HW accepts torch.tensor or numpy.array
        """
        raise NotImplemented("TODO Add the COLOR SCHEMA CORRECTLY")
        img = self.plot_image(img, not_log=True)
        try:
            label = label.clone().cpu().numpy()
        except:
            pass
        label = label.astype(np.long)

        H, W, C = img.shape
        uni = np.unique(label)
        overlay = np.zeros_like(img)

        centers = []
        for u in uni:
            m = label == u
            col = SCANNET_COLORS[u]
            overlay[m] = col
            labels_mask = skimage.measure.label(m)
            regions = skimage.measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            cen = np.mean(regions[0].coords, axis=0).astype(np.uint32)[::-1]

            centers.append((SCANNET_CLASSES[u], cen))

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore)))
        draw = ImageDraw.Draw(img_new)

        if not text_off:

            for i in centers:
                pose = i[1]
                pose[0] -= len(str(i[0])) * shift
                pose[1] -= font_size / 2
                draw.text(tuple(pose), str(i[0]), fill=(255, 255, 255, 128))

        img_new = img_new.convert("RGB")
        mask = skimage.segmentation.mark_boundaries(np.array(img_new), label, color=(255, 255, 255))
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)
