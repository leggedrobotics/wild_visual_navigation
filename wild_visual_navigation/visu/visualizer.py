# TODO: Jonas doc strings, rework visualiation functions

import os
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
from wild_visual_navigation.visu import image_functionality
import torch
import skimage
import seaborn as sns
import pytorch_lightning as pl
from typing import Optional
from wild_visual_navigation.learning.utils import get_confidence

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
        self._c_maps = {"RdYlBu": np.array([np.uint8(np.array(c) * 255) for c in sns.color_palette("RdYlBu", 256)])}

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

    def plot_mission_node_prediction(self, node: any):
        if node._image is None or node._prediction is None:
            return None

        trav_pred = node._prediction[:, 0]
        reco_pred = node._prediction[:, 1:]
        conf_pred = get_confidence(reco_pred, node._features)

        trav_img = self.plot_traversability_graph_on_seg(
            trav_pred,
            node.feature_segments,
            node.as_pyg_data(),
            node.feature_positions,
            node.image.clone(),
            colormap="RdYlBu",
        )
        conf_img = self.plot_traversability_graph_on_seg(
            conf_pred,
            node.feature_segments,
            node.as_pyg_data(),
            node.feature_positions,
            node.image.clone(),
            colormap="RdYlBu",
        )
        return trav_img, conf_img

    def plot_mission_node_training(self, node: any):
        if node._image is None or node._prediction is None:
            return None

        if node.supervision_signal is None:
            sa = node.image.shape
            supervison_img = np.zeros((sa[1], sa[2], sa[0]), dtype=np.uint8)
        else:
            supervison_img = self.plot_traversability_graph(
                node.supervision_signal,
                node.as_pyg_data(),
                node.feature_positions,
                node.image.clone(),
                colormap="RdYlBu",
            )

        mask = torch.isnan(node.supervision_mask)
        supervision_mask = node.supervision_mask.clone()
        supervision_mask[mask] = 0
        mask_img = self.plot_detectron(
            node.image.clone(),
            torch.round(torch.clamp(255 * supervision_mask[0], 0, 255)),
            max_seg=256,
            colormap="RdYlBu",
            overlay_mask=mask[0],
            draw_bound=False,
        )

        return supervison_img, mask_img

    @image_functionality
    def plot_traversability_graph_on_seg(
        self, prediction, seg, graph, center, img, max_val=1.0, colormap="RdYlBu", **kwargs
    ):

        # Transfer the node traversbility score to the segment
        m = torch.zeros_like(seg, dtype=prediction.dtype)
        for i in range(seg.max() + 1):
            m[seg == i] = prediction[i]

        # Plot Segments on Image
        i1 = self.plot_detectron_cont(
            img.detach().cpu().numpy(), m.detach().cpu().numpy(), not_log=True, colormap=colormap
        )
        i2 = (torch.from_numpy(i1).type(torch.float32) / 255).permute(2, 0, 1)
        # Plot Graph on Image
        return self.plot_traversability_graph(prediction, graph, center, i2, not_log=True, colormap=colormap)

    @image_functionality
    def plot_traversability_graph(self, prediction, graph, center, img, max_val=1.0, colormap="RdYlBu", **kwargs):
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
        prediction = (prediction.type(torch.float32) * 255).type(torch.uint8)
        img = np.uint8((img.permute(1, 2, 0) * 255).cpu().numpy())

        img_pil = Image.fromarray(img)
        img_draw = ImageDraw.Draw(img_pil)

        # Get attributes from the graph
        adjacency_list = graph.edge_index

        try:
            node_validity = graph.y_valid
        except Exception:
            node_validity = [True] * adjacency_list.shape[1]

        if colormap not in self._c_maps:
            self._c_maps[colormap] = np.array([np.uint8(np.array(c) * 255) for c in sns.color_palette(colormap, 256)])
        c_map = self._c_maps[colormap]

        for i in range(adjacency_list.shape[1]):
            a, b = adjacency_list[0, i], adjacency_list[1, i]
            line_params = center[a].tolist() + center[b].tolist()
            img_draw.line(line_params, fill=(127, 127, 127))

        for i in range(center.shape[0]):
            params = center[i].tolist()
            params = [p - elipse_size for p in params] + [p + elipse_size for p in params]
            if node_validity[i]:
                img_draw.ellipse(params, width=10, fill=tuple(c_map[prediction[i]].tolist()))
            else:
                img_draw.ellipse(params, width=1, fill=(127, 127, 127))

        return np.array(img_pil)

    @image_functionality
    def plot_detectron(
        self, img, seg, alpha=0.5, draw_bound=True, max_seg=40, colormap="Set2", overlay_mask=None, **kwargs
    ):
        img = self.plot_image(img, not_log=True)
        # assert seg.max() < max_seg and seg.min() >= 0, f"Seg out of Bounds: 0-{max_seg}, Given: {seg.min()}-{seg.max()}"
        try:
            np_seg = seg.clone().cpu().numpy()
        except Exception:
            pass
        np_seg = np_seg.astype(np.uint32)

        H, W, C = img.shape
        overlay = np.zeros_like(img)
        c_map = sns.color_palette(colormap, max_seg)

        uni = np.unique(np_seg)
        # Commented out center extraction code
        # centers = []
        for u in uni:
            m = np_seg == u
            try:
                col = np.uint8(np.array(c_map[u])[:3] * 255)
            except Exception as e:
                print(e)
                continue
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
        if overlay_mask is not None:
            try:
                overlay_mask = overlay_mask.cpu().numpy()
            except Exception:
                pass
            fore[overlay_mask] = 0

        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore)))

        img_new = img_new.convert("RGB")
        mask = skimage.segmentation.mark_boundaries(np.array(img_new), np_seg, color=(255, 255, 255))
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)

    @image_functionality
    def plot_detectron_cont(self, img, seg, alpha=0.3, max_val=1.0, colormap="RdYlBu", **kwargs):
        img = self.plot_image(img, not_log=True)
        assert (
            seg.max() <= max_val and seg.min() >= 0
        ), f"Seg out of Bounds: 0-{max_val}, Given: {seg.min()}-{seg.max()}"
        try:
            seg = seg.clone().cpu().numpy()
        except Exception:
            pass
        seg = np.uint8(seg.astype(np.float32) * 255)

        H, W, C = img.shape
        overlay = np.zeros_like(img)

        if colormap not in self._c_maps:
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
    def plot_graph_result(self, graph, center, img, seg, res, use_seg=False, colormap="RdYlBu", **kwargs):
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
        except Exception:
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
        except Exception:
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
    def plot_optical_flow(self, flow: torch.Tensor, img1: torch.Tensor, img2: torch.Tensor, s=10, **kwargs):
        """Draws line connection between to images based on estimated flow

        Args:
            flow (torch.Tensor,  dtype=torch.float32, shape=(2,H,W)): Flow
            img1 ((torch.Tensor,  dtype=torch.float32, shape=(C,H,W) or (H,W,C)): Img1
            img2 (torch.Tensor,  dtype=torch.float32, shape=(C,H,W) or (H,W,C)): Img2
        Returns:
            (np.array, dtype=np.uint8, shape=(H,2xW,C)): Concatenated image with flow lines
        """

        i1 = self.plot_image(img1, not_log=True, store=False)
        i2 = self.plot_image(img2, not_log=True, store=False)
        img = np.concatenate([i1, i2], axis=1)

        pil_img = Image.fromarray(img, "RGB")
        draw = ImageDraw.Draw(pil_img)

        col = (0, 255, 0)
        for u in range(int(flow.shape[1] / s)):
            u = int(u * s)
            for v in range(int(flow.shape[2] / s)):
                v = int(v * s)
                du = (flow[0, u, v]).item()
                dv = (flow[1, u, v] + i2.shape[1]).item()
                try:
                    draw.line([(v, u), (v + dv, u + du)], fill=col, width=2)
                except:
                    pass
        return np.array(pil_img).astype(np.uint8)

    @image_functionality
    def plot_sparse_optical_flow(
        self, pre_pos: torch.Tensor, cur_pos: torch.Tensor, img1: torch.Tensor, img2: torch.Tensor, **kwargs
    ):
        """Draws line connection between to images based on estimated flow

        Args:
            pre_pos (torch.Tensor,  dtype=torch.float32, shape=(N,2)): Flow
            cur_pos (torch.Tensor,  dtype=torch.float32, shape=(N,2)): Flow
            img1 ((torch.Tensor,  dtype=torch.float32, shape=(C,H,W) or (H,W,C)): Img1
            img2 (torch.Tensor,  dtype=torch.float32, shape=(C,H,W) or (H,W,C)): Img2
        Returns:
            (np.array, dtype=np.uint8, shape=(H,2xW,C)): Concatenated image with flow lines
        """

        i1 = self.plot_image(img1, not_log=True, store=False)
        i2 = self.plot_image(img2, not_log=True, store=False)
        img = np.concatenate([i1, i2], axis=1)

        pil_img = Image.fromarray(img, "RGB")
        draw = ImageDraw.Draw(pil_img)

        col = (0, 255, 0)
        for p, c in zip(pre_pos, cur_pos):
            try:
                draw.line([(p[0].item(), p[1].item()), ((i2.shape[1] + c[0]).item(), c[1].item())], fill=col, width=2)
            except:
                pass
        return np.array(pil_img).astype(np.uint8)

    @image_functionality
    def plot_corrospondence_segment(
        self,
        seg_prev: torch.Tensor,
        seg_current: torch.Tensor,
        img_prev: torch.Tensor,
        img_current: torch.Tensor,
        center_prev: torch.Tensor,
        center_current: torch.Tensor,
        corrospondence: torch.Tensor,
        **kwargs,
    ):
        """_summary_

        Args:
            seg_prev (torch.Tensor, dtype=torch.long, shape=(H,W))): Segmentation
            seg_current (torch.Tensor, dtype=torch.long, shape=(H,W)): Segmentation
            img_prev (torch.Tensor,  dtype=torch.float32, shape=(C,H,W) or (H,W,C)): Image
            img_current (torch.Tensor,  dtype=torch.float32, shape=(C,H,W) or (H,W,C)): Image
            center_prev (torch.Tensor,  dtype=torch.float32, shape=(N,2)): Center positions to index seg reverse order
            center_current (torch.Tensor,  dtype=torch.float32, shape=(N,2)): Center positions to index seg reverse order
            corrospondence (torch.Tensor,  dtype=torch.long, shape=(N,2)): 0 previous seg, 1 current seg

        Returns:
            (np.array, dtype=np.uint8, shape=(H,2xW,C)): Concatenated image with segments and connected centers
        """
        prev_img = self.plot_detectron(img_prev, seg_prev, max_seg=seg_prev.max() + 1, not_log=True, store=False)
        current_img = self.plot_detectron(
            img_current, seg_current, max_seg=seg_current.max() + 1, not_log=True, store=False
        )
        img = np.concatenate([prev_img, current_img], axis=1)

        pil_img = Image.fromarray(img, "RGB")
        draw = ImageDraw.Draw(pil_img)
        col = (0, 255, 0)
        for cp, cc in zip(center_prev[corrospondence[:, 0]], center_current[corrospondence[:, 1]]):
            try:
                draw.line(
                    [(cp[0].item(), cp[1].item()), (cc[0].item() + img_prev.shape[1], cc[1].item())], fill=col, width=2
                )
            except:
                pass
        return np.array(pil_img).astype(np.uint8)


if __name__ == "__main__":
    # Data was generated in the visu function of the lightning_module with the following code
    # from PIL import Image
    # import numpy as np
    # Image.fromarray(np.uint8(img[b].cpu().numpy().transpose(1,2,0)*255)).save("assets/graph/img.png")
    # torch.save( graph[b], "assets/graph/graph.pt")
    # torch.save( center[b], "assets/graph/center.pt")
    # torch.save( pred[:, 0], "assets/graph/trav_pred.pt")
    # torch.save( pred[:, 1:], "assets/graph/reco_pred.pt")
    # torch.save( seg[b], "assets/graph/seg.pt")

    from wild_visual_navigation import WVN_ROOT_DIR

    img = np.array(Image.open(os.path.join(WVN_ROOT_DIR, "assets/graph/img.png")))
    img = (torch.from_numpy(img).type(torch.float32) / 255).permute(2, 0, 1)

    graph = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/graph.pt"))
    center = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/center.pt"))
    trav_pred = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/trav_pred.pt"))
    reco_pred = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/reco_pred.pt"))
    seg = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/seg.pt"))
    conf = get_confidence(reco_pred, graph.x)

    visu = LearningVisualizer(p_visu=os.path.join(WVN_ROOT_DIR, "results/test_visu"))

    print("Plot Image: CHW", img.shape, img.dtype, type(img))
    visu.plot_image(img=img, store=True, tag="1")
    print("Plot Image: HWC", img.permute(1, 2, 0).shape, img.dtype, type(img))
    visu.plot_image(img=img.permute(1, 2, 0), store=True, tag="2")

    # seg = np.random.randint( 0, 100, (400,400) )
    print("plot_segmentation: HW", seg.shape, seg.dtype, type(seg))
    visu.plot_segmentation(seg=seg, store=True, max_seg=100, tag="3")
    print("plot_segmentation: CHW", seg[None].shape, seg.dtype, type(seg))
    visu.plot_segmentation(seg=seg[None], max_seg=100, store=True, tag="4")

    print("plot_segmentation: HW", seg.shape, seg.dtype, type(seg))
    visu.plot_detectron(img=img, seg=seg, store=True, max_seg=100, tag="5")

    print("plot_segmentation_count: HW", seg.shape, seg.dtype, type(seg))
    seg = seg.type(torch.float32) / seg.max()
    visu.plot_detectron_cont(img=img, seg=seg, store=True, tag="6")

    i1 = visu.plot_traversability_graph(trav_pred, graph, center, img, not_log=True)
    i2 = visu.plot_traversability_graph(graph.y, graph, center, img, not_log=True)
    visu.plot_list(imgs=[i1, i2], tag="7_Trav_Graph_only", store=True)

    seg = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/seg.pt"))
    # Visualize Graph with Segmentation
    i1 = visu.plot_traversability_graph_on_seg(trav_pred, seg, graph, center, img, not_log=True)
    i2 = visu.plot_traversability_graph_on_seg(graph.y, seg, graph, center, img, not_log=True)
    i3 = visu.plot_image(img, not_log=True)
    visu.plot_list(imgs=[i1, i2, i3], tag="8_Trav", store=True)

    i1 = visu.plot_traversability_graph_on_seg(conf, seg, graph, center, img, not_log=True)
    i2 = visu.plot_traversability_graph_on_seg(graph.y_valid.type(torch.float32), seg, graph, center, img, not_log=True)
    i3 = visu.plot_image(img, not_log=True)
    visu.plot_list(imgs=[i1, i2, i3], tag="9_Confidence", store=True)
