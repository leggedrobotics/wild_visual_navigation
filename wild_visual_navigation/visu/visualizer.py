# TODO: Jonas doc strings, rework visualiation functions

import os
import numpy as np
import imageio
import cv2
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import cm
from wild_visual_navigation.utils import get_img_from_fig
from wild_visual_navigation.visu import image_functionality
import torch
import skimage
import seaborn as sns
from torch_geometric.data import Data
import pytorch_lightning as pl
from wild_visual_navigation.traversability_estimator import MissionNode

__all__ = ["LearningVisualizer"]


class LearningVisualizer:
    def __init__(
        self,
        p_visu: Optional[str] = None,
        store: Optional[bool] = False,
        pl_model: Optional[pl.LightningModule] = None,
        epoch: int = 0,
        log: bool = False,
    ):
        self._p_visu = p_visu
        self._pl_model = pl_model
        self._epoch = epoch
        self._store = store
        self._log = log
        self._c_maps = {"viridis": np.array([np.uint8(np.array(c) * 255) for c in sns.color_palette("viridis", 256)])}

        if not (p_visu is None):
            if not os.path.exists(self._p_visu):
                os.makedirs(self._p_visu)
        else:
            self._store = False

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def store(self) -> bool:
        return self._store

    @store.setter
    def store(self, store: bool):
        self._store = store

    @image_functionality
    def plot_list(self, imgs, **kwargs):
        return np.concatenate(imgs, axis=1)

    def plot_mission_node_training(self, node: MissionNode):
        # TODO: implement to draw the segmentation and uncertainty
        if node._image is None or node._supervision_mask is None:
            return None
        img_np = kornia.utils.tensor_to_image(node._image)
        trav_np = kornia.utils.tensor_to_image(node._supervision_mask)

        # Draw segments
        img_np = segmentation.mark_boundaries(img_np, node._feature_segments.cpu().numpy(), color=(0.7, 0.7, 0.7))

        img_pil = Image.fromarray(np.uint8(img_np * 255))
        img_draw = ImageDraw.Draw(img_pil)
        trav_pil = Image.fromarray(np.uint8(trav_np * 255))

        # Draw graph
        for i in range(node._feature_edges.shape[1]):
            a, b = node._feature_edges[0, i], node._feature_edges[1, i]
            line_params = node._feature_positions[a].tolist() + node._feature_positions[b].tolist()
            img_draw.line(line_params, fill=(255, 255, 255, 100), width=2)

        for i in range(node._feature_positions.shape[0]):
            params = node._feature_positions[i].tolist()
            color = trav_pil.getpixel((params[0], params[1]))
            r = 5
            params = [p - r for p in params] + [p + r for p in params]
            img_draw.ellipse(params, fill=color)

        np_draw = np.array(img_pil)
        return np_draw.astype(np.uint8)

    def plot_mission_node_prediction(self, node: MissionNode):
        # TODO: convert to visu function
        if node._image is None or node._prediction is None:
            return None
        img_np = kornia.utils.tensor_to_image(node._image) * 255
        seg_np = kornia.utils.tensor_to_image(node._feature_segments)
        seg_np_rgb = img_np.copy()
        confidence_np_rgb = img_np.copy()

        colormap = cm.get_cmap("autumn", 255)
        col_map = np.uint8(colormap(np.linspace(0, 1, 256))[:, :3] * 255)

        # TODO: implement to draw the segmentation and uncertainty
        max_seg = 1.0
        y_pred = (node._prediction.clip(0, max_seg) / max_seg * 255).type(torch.uint8).cpu()
        label_pred = y_pred[:, 0]
        confidence_pred = (torch.sigmoid(y_pred[:, 1:]) - node._features.cpu()).abs().sum(dim=1)
        confidence_pred -= confidence_pred.min()
        confidence_pred /= confidence_pred.max()
        confidence_pred = 1.0 - confidence_pred
        confidence_pred = (confidence_pred.clip(0, max_seg) / max_seg * 255).type(torch.uint8)

        # segments to CPU
        seg = node._feature_segments.cpu()

        # iterate segments
        for i in range(node._feature_positions.shape[0]):
            params = node._feature_positions[i].tolist()
            segment_id = node._feature_segments[int(params[0]), int(params[1])].item()

            seg_np_rgb[seg == segment_id] = tuple(col_map[label_pred[i]].tolist())
            confidence_np_rgb[seg == segment_id] = tuple(col_map[confidence_pred[i]].tolist())

        # blending
        alpha = 0.5
        seg_np_rgb = alpha * img_np + (1 - alpha) * seg_np_rgb
        confidence_np_rgb = alpha * img_np + (1 - alpha) * confidence_np_rgb

        return seg_np_rgb.astype(np.uint8), confidence_np_rgb.astype(np.uint8)

    @image_functionality
    def plot_traversability_graph_on_seg(
        self, prediction, seg, graph, center, img, max_val=1.0, colormap="viridis", **kwargs
    ):

        # Transfer the node traversbility score to the segment
        m = torch.zeros_like(seg, dtype=prediction.dtype)
        for i in range(seg.max() + 1):
            m[seg == i] = prediction[i]

        # Plot Segments on Image
        i1 = self.plot_detectron_cont(img.detach().cpu().numpy(), m.detach().cpu().numpy(), not_log=True)
        i2 = (torch.from_numpy(i1).type(torch.float32) / 255).permute(2, 0, 1)
        # Plot Graph on Image
        return self.plot_traversability_graph(prediction, graph, center, i2, not_log=True)

    @image_functionality
    def plot_traversability_graph(self, prediction, graph, center, img, max_val=1.0, colormap="viridis", **kwargs):
        """Plot prediction on graph

        Args:
            prediction (torch.Tensor,  dtype=torch.float32, shape=(N)): Perdicted label
            graph (torch_geometric.data.data.Data): Graph
            center (torch.Tensor,  dtype=torch.float32, shape=(1, N, 2)): Center positions in image plane
            img (torch.Tensor,  dtype=torch.float32, shape=(3, H, W)): Image
            seg (torch.Tensor,  dtype=torch.int64, shape=(H, W)): Segmentation mask
        Returns:
            img (np.Array,  dtype=np.uint8, shape=(3, H, W)): Left prediction, right ground truth
        """
        assert (
            prediction.max() <= max_val and prediction.min() >= 0
        ), f"Pred out of Bounds: 0-1, Given: {prediction.min()}-{prediction.max()}"
        elipse_size = 5
        prediction = (prediction.type(torch.float32) / 255).type(torch.uint8)
        img = np.uint8((img.permute(1, 2, 0) * 255).cpu().numpy())

        img_pil = Image.fromarray(img)
        img_draw = ImageDraw.Draw(img_pil)

        adjacency_list = graph.edge_index

        if not colormap in self._c_maps:
            self._c_maps[colormap] = np.array([np.uint8(np.array(c) * 255) for c in sns.color_palette(colormap, 256)])
        c_map = self._c_maps[colormap]

        for i in range(adjacency_list.shape[1]):
            a, b = adjacency_list[0, i], adjacency_list[1, i]
            line_params = center[a].tolist() + center[b].tolist()
            img_draw.line(line_params, fill=(127, 127, 127))

        for i in range(center.shape[0]):
            params = center[i].tolist()
            params = [p - elipse_size for p in params] + [p + elipse_size for p in params]
            img_draw.ellipse(params, width=10, fill=tuple(c_map[prediction[i]].tolist()))

        return np.array(img_pil)

    @image_functionality
    def plot_detectron(self, img, seg, alpha=0.5, draw_bound=True, max_seg=40, colormap="Set2", **kwargs):
        img = self.plot_image(img, not_log=True)
        assert seg.max() < max_seg and seg.min() >= 0, f"Seg out of Bounds: 0-{max_seg}, Given: {seg.min()}-{seg.max()}"
        try:
            seg = seg.clone().cpu().numpy()
        except:
            pass
        seg = seg.astype(np.uint32)

        H, W, C = img.shape
        overlay = np.zeros_like(img)
        c_map = sns.color_palette(colormap, max_seg)

        uni = np.unique(seg)
        # Commented out center extraction code
        # centers = []
        for u in uni:
            m = seg == u
            col = np.uint8(np.array(c_map[u])[:3] * 255)
            overlay[m] = col
            # segs_mask = skimage.measure.label(m)
            # regions = skimage.measure.regionprops(segs_mask)
            # regions.sort(key=lambda x: x.area, reverse=True)
            # cen = np.mean(regions[0].coords, axis=0).astype(np.uint32)[::-1]
            # centers.append((self._meta_data["stuff_classes"][u], cen))

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore)))

        img_new = img_new.convert("RGB")
        mask = skimage.segmentation.mark_boundaries(np.array(img_new), seg, color=(255, 255, 255))
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)

    @image_functionality
    def plot_detectron_cont(self, img, seg, alpha=0.5, max_val=1.0, colormap="viridis", **kwargs):
        img = self.plot_image(img, not_log=True)
        assert (
            seg.max() <= max_val and seg.min() >= 0
        ), f"Seg out of Bounds: 0-{max_val}, Given: {seg.min()}-{seg.max()}"
        try:
            seg = seg.clone().cpu().numpy()
        except:
            pass
        seg = np.uint8(seg.astype(np.float32) * 255)

        H, W, C = img.shape
        overlay = np.zeros_like(img)

        if not colormap in self._c_maps:
            self._c_maps[colormap] = np.array([np.uint8(np.array(c) * 255) for c in sns.color_palette(colormap, 256)])
        c_map = self._c_maps[colormap]

        uni = np.unique(seg)
        for u in uni:
            m = seg == u
            overlay[m] = c_map[u]

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore)))
        img_new = np.array(img_new.convert("RGB"))
        return np.uint8(img_new)

    @image_functionality
    def plot_graph_result(self, graph, center, img, seg, res, use_seg=False, colormap="viridis", **kwargs):
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

        viridis = cm.get_cmap(colormap, 255)
        c_map = np.uint8(viridis(np.linspace(0, 1, 256))[:, :3] * 255)

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
            img_draw.ellipse(params, width=10, fill=tuple(c_map[y[i]].tolist()))
            img_pred_draw.ellipse(params, width=10, fill=tuple(c_map[y_pred[i]].tolist()))
            if use_seg:
                seg_draw.ellipse(params, width=10, fill=tuple(c_map[y[i]].tolist()))

        img_res = np.array(img_pil)
        img_pred_res = np.array(img_pred_pil)

        ls = [img_pred_res, img_res]
        if use_seg:
            ls = ls + [np.array(seg_pil)]
        res = np.concatenate(ls, axis=1)
        return res

    @image_functionality
    def plot_segmentation(self, seg, max_seg=40, colormap="Set2", **kwargs):
        try:
            seg = seg.clone().cpu().numpy()
        except:
            pass

        if seg.shape[0] == 1:
            seg = seg[0]

        if seg.dtype == bool:
            max_seg = 2

        c_map = sns.color_palette(colormap, max_seg)

        H, W = seg.shape[:2]
        img = np.zeros((H, W, 3), dtype=np.uint8)

        uni = np.unique(seg)

        for u in uni:
            img[seg == u] = np.uint8(np.array(c_map[u])[:3] * 255)

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


if __name__ == "__main__":
    from wild_visual_navigation import WVN_ROOT_DIR

    visu = LearningVisualizer(p_visu=os.path.join(WVN_ROOT_DIR, "results/test_visu"))
    img = np.array(Image.open(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png")))
    seg = np.zeros_like(img[:, :, 0], dtype=np.long)
    seg[:100, :100] = 4
    seg[300:500, 200:500] = 20
    seg[300:500, 600:1440] = 30
    seg[400:1080, 640:900] = 35

    # print("Plot Image: HWC", img.shape, img.dtype, type(img))
    # visu.plot_image(img=img, store=True, tag="1")
    # print("Plot Image: CHW", img.transpose(2,0,1).shape, img.dtype, type(img))
    # visu.plot_image(img=img.transpose(2,0,1), store=True, tag="2")

    # # seg = np.random.randint( 0, 100, (400,400) )
    # print("plot_segmentation: HW", seg.shape, seg.dtype, type(seg))
    # visu.plot_segmentation(seg=seg, store=True, max_seg=40,  tag="3")
    # print("plot_segmentation: CHW", seg[None].shape, seg.dtype, type(seg))
    # visu.plot_segmentation(seg=seg[None], max_seg=40, store=True, tag="4")

    # print("plot_segmentation: HW", seg.shape, seg.dtype, type(seg))
    # visu.plot_detectron(img=img, seg=seg, store=True, max_seg=40,  tag="5")

    print("plot_segmentation_cont: HW", seg.shape, seg.dtype, type(seg))
    seg = seg.astype(np.float32) / seg.max()
    visu.plot_detectron_cont(img=img, seg=seg, store=True, tag="6")
