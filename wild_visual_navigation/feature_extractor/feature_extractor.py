from wild_visual_navigation.feature_extractor import (
    StegoInterface,
    DinoInterface,
    SegmentExtractor,
    TorchVisionInterface,
)
from pytictac import Timer
import skimage
import torch
import numpy as np
import kornia
from kornia.feature import DenseSIFTDescriptor
from kornia.contrib import extract_tensor_patches, combine_tensor_patches
from torchvision import transforms as T
from PIL import Image, ImageDraw


class FeatureExtractor:
    def __init__(
        self, device: str, segmentation_type: str = "slic", feature_type: str = "dino", input_size: int = 448, **kwargs
    ):
        """Feature extraction from image

        Args:
            device (str): Compute device
            extractor (str): Extractor model: stego, dino_slic
        """

        self._device = device
        self._segmentation_type = segmentation_type
        self._feature_type = feature_type
        self._input_size = input_size
        # Prepare segment extractor
        self.segment_extractor = SegmentExtractor().to(self._device)

        # Prepare extractor depending on the type
        if self._feature_type == "stego":
            self._feature_dim = 90
            self.extractor = StegoInterface(device=device, input_size=input_size)
        elif self._feature_type == "dino":
            self._feature_dim = 90

            self.extractor = DinoInterface(device=device, input_size=input_size, patch_size=kwargs.get("patch_size", 8))
        elif self._feature_type == "sift":
            self._feature_dim = 128
            self.extractor = DenseSIFTDescriptor().to(device)
        elif self._feature_type == "torchvision":
            self._extractor = TorchVisionInterface(
                device=device, model_type=kwargs["model_type"], input_size=input_size
            )
        elif self._feature_type == "histogram":
            self._feature_dim = 90
        elif self._feature_type == "none":
            pass
        else:
            raise f"Extractor[{self._feature_type}] not supported!"

        if self.segmentation_type == "slic":
            from fast_slic import Slic

            self.slic_num_components = kwargs.get("slic_num_components", 100)
            self.slic = Slic(
                num_components=kwargs.get("slic_num_components", 100), compactness=kwargs.get("slic_compactness", 10)
            )

        elif self.segmentation_type == "random":
            pass

    def extract(self, img, **kwargs):
        if self._segmentation_type == "random":
            dense_feat = self.compute_features(img, None, None, **kwargs)

            H, W = img.shape[2:]
            nr = kwargs.get("n_random_pixels", 100)
            seg = torch.full((H * W,), -1, dtype=torch.long, device=self._device)
            indices = torch.randperm(H * W, device=self._device)[:nr]
            seg[indices] = torch.arange(0, nr, device=self._device)
            seg = seg.reshape(H, W)
            feat = dense_feat[0].reshape(dense_feat.shape[1], H * W)[:, indices].T

            if kwargs.get("return_dense_features", False):
                return None, feat, seg, None, dense_feat

            return None, feat, seg, None

        # Compute segments, their centers, and edges connecting them (graph structure)
        with Timer("feature_extractor - compute_segments"):
            edges, seg, center = self.compute_segments(img, **kwargs)

        # Compute features
        with Timer("feature_extractor - compute_features"):
            dense_feat = self.compute_features(img, seg, center, **kwargs)

        with Timer("feature_extractor - compute_features"):
            # Sparsify features to match the centers if required
            feat = self.sparsify_features(dense_feat, seg)

        if kwargs.get("return_dense_features", False):
            return edges, feat, seg, center, dense_feat

        return edges, feat, seg, center

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def segmentation_type(self):
        return self._segmentation_type

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._device = device
        self.extractor.change_device(device)

    def compute_segments(self, img: torch.tensor, **kwargs):
        if self._segmentation_type == "none" or self._segmentation_type is None:
            edges, seg, centers = self.segment_pixelwise(img, **kwargs)

        elif self._segmentation_type == "grid":
            seg = self.segment_grid(img, **kwargs)

        elif self._segmentation_type == "slic":
            seg = self.segment_slic(img, **kwargs)

        elif self._segmentation_type == "stego":
            seg = self.segment_stego(img, **kwargs)

        elif self._segmentation_type == "random":
            seg = self.segment_random(img, **kwargs)

        else:
            raise f"segmentation_type [{self._segmentation_type}] not supported"

        # Compute edges and centers
        if self._segmentation_type != "none" and self._segmentation_type is not None:
            # Extract adjacency_list based on segments
            edges = self.segment_extractor.adjacency_list(seg[None, None])
            # Extract centers
            centers = self.segment_extractor.centers(seg[None, None])

        return edges.T, seg, centers

    def segment_pixelwise(self, img, **kwargs):
        # Generate pixel-wise segmentation
        B, C, H, W = img.shape
        seg = torch.arange(0, H * W, 1).reshape(H, W).to(self._device)

        # Centers
        seg_x = torch.ones((1, 1, H, W), dtype=torch.int32).to(self._device)
        seg_y = torch.ones((1, 1, H, W), dtype=torch.int32).to(self._device)
        seg_x = (seg_x.cumsum(dim=-1) - 1).reshape(-1, 1)
        seg_y = (seg_y.cumsum(dim=-2) - 1).reshape(-1, 1)
        centers = torch.cat((seg_y, seg_x), dim=1)

        # Edges
        hor_edges = torch.cat((seg[:, :-1].reshape(-1, 1), seg[:, 1:].reshape(-1, 1)), dim=1)
        ver_edges = torch.cat((seg[:-1, :].reshape(-1, 1), seg[1:, :].reshape(-1, 1)), dim=1)
        edges = torch.cat((hor_edges, ver_edges), dim=0)

        return edges, seg, centers

    def segment_grid(self, img, **kwargs):
        cell_size = kwargs.get("cell_size", 32)
        patch_size = (cell_size, cell_size)

        B, C, H, W = img.shape
        patches = extract_tensor_patches(
            input=torch.ones((1, 1, H, W), dtype=torch.int32).to(self._device),
            window_size=patch_size,
            stride=patch_size,
        )
        for i in range(patches.shape[1]):
            patches[:, i, :, :, :] = i

        combine_patch_size = (int(H / cell_size), int(W / cell_size))
        seg = combine_tensor_patches(
            patches=patches, original_size=(H, W), window_size=combine_patch_size, stride=combine_patch_size
        )

        return seg[0, 0].to(self._device)

    def segment_slic(self, img, **kwargs):
        # Get slic clusters
        img_np = kornia.utils.tensor_to_image(img)
        seg = self.slic.iterate(np.uint8(np.ascontiguousarray(img_np) * 255))
        seg = torch.from_numpy(seg).to(self._device).type(torch.long)
        # TODO verify to be unnecessary        
        # if self.slic_num_components > kwargs.get("n_segments", 100):
        #     unique = torch.unique(seg)
        #     unique = torch.randperm(unique)[: min(kwargs.get("n_segments", 100), unique.shape[0])]
        #     H, W = seg.shape
        #     seg = seg.reshape(-1)
        #     seg[unique] = -1
        #     seg.reshape(H, W)
        return seg

    def segment_random(self, img, **kwargs):
        # Randomly select
        H, W = img.shape[2:]
        nr = kwargs.get("n_random_pixels", 100)
        seg = torch.full((H * W,), -1, dtype=torch.long, device=self._device)
        indices = torch.randperm(H * W, device=self._device)[:nr]
        seg[indices] = torch.arange(0, nr, device=self._device)
        seg = seg.reshape(H, W)
        return seg

    def segment_stego(self, img, **kwargs):
        # Prepare input image
        img_internal = img.clone()
        self.extractor.inference_crf(img_internal)
        seg = torch.from_numpy(self.extractor.cluster_segments).to(self._device)

        # Change the segment indices by numbers from 0 to N
        for i, k in enumerate(seg.unique()):
            seg[seg == k.item()] = i

        return seg

    def compute_features(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        if self._feature_type == "histogram":
            feat = self.compute_histogram(img, seg, center, **kwargs)

        elif self._feature_type == "sift":
            feat = self.compute_sift(img, seg, center, **kwargs)

        elif self._feature_type == "dino":
            feat = self.compute_dino(img, seg, center, **kwargs)

        elif self._feature_type == "stego":
            feat = self.compute_stego(img, seg, center, **kwargs)

        elif self._feature_type == "torchvision":
            feat = self.compute_torchvision(img, seg, center, **kwargs)
        elif self._feature_type == "none":
            feat = None
        else:
            raise f"segmentation_type [{self._segmentation_type}] not supported"

        return feat

    def compute_histogram(self, img: torch.tensor, seg: torch.tensor, **kwargs):
        raise NotImplementedError("compute_histogram is not implemented yet")

    @torch.no_grad()
    def compute_sift(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        B, C, H, W = img.shape
        if C == 3:
            feat_r = self.extractor(img[:, 0, :, :][None])
            feat_g = self.extractor(img[:, 1, :, :][None])
            feat_b = self.extractor(img[:, 2, :, :][None])
            features = torch.cat([feat_r, feat_r, feat_b], dim=1)
        else:
            features = self.extractor(img)
        return features

    @torch.no_grad()
    def compute_dino(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        img_internal = img.clone()
        features = self.extractor.inference(img_internal)
        return features

    @torch.no_grad()
    def compute_torchvision(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        img_internal = img.clone()
        features = self._extractor.inference(img_internal)
        return features

    @torch.no_grad()
    def compute_stego(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        return self.extractor.features

    def sparsify_features(self, dense_features: torch.tensor, seg: torch.tensor, cumsum_trick=False):
        if self._feature_type not in ["histogram"] and self._segmentation_type not in ["none"]:
            # Get median features for each cluster

            if type(dense_features) == dict:
                # Multiscale feature pyramid extraction
                scales_x = [feat.shape[2] / seg.shape[0] for feat in dense_features.values()]
                scales_y = [feat.shape[3] / seg.shape[1] for feat in dense_features.values()]

                segs = [
                    torch.nn.functional.interpolate(
                        seg[None, None, :, :].type(torch.float32), scale_factor=(scale_x, scale_y)
                    )[0, 0].type(torch.long)
                    for scale_x, scale_y in zip(scales_x, scales_y)
                ]
                sparse_features = []

                # Iterate over each segment
                for i in range(seg.max() + 1):
                    single_segment_feature = []

                    # Iterate over each scale
                    for dense_feature, seg_scaled in zip(dense_features.values(), segs):
                        m = seg_scaled == i
                        # When downscaling the mask it becomes 0 therfore calculate x,y
                        # Based on the previous scale
                        if m.sum() == 0:
                            x = (
                                (prev_x * seg_scaled.shape[0] / prev_scale_x)
                                .type(torch.long)
                                .clamp(0, seg_scaled.shape[0] - 1)
                            )
                            y = (
                                (prev_y * seg_scaled.shape[1] / prev_scale_y)
                                .type(torch.long)
                                .clamp(0, seg_scaled.shape[1] - 1)
                            )
                            feat = dense_feature[0, :, x, y]
                        else:
                            x, y = torch.where(m)
                            prev_x = x.type(torch.float32).mean()
                            prev_y = y.type(torch.float32).mean()
                            prev_scale_x = seg_scaled.shape[0]
                            prev_scale_y = seg_scaled.shape[0]
                            feat = dense_feature[0, :, x, y].mean(dim=1)

                        single_segment_feature.append(feat)

                    single_segment_feature = torch.cat(single_segment_feature, dim=0)
                    sparse_features.append(single_segment_feature)
                return torch.stack(sparse_features, dim=1).T

            else:
                if cumsum_trick:
                    # Cumsum is slightly slower for 100 segments
                    # Trick: sort the featuers according to the segments and then use cumsum for summing
                    dense_features = dense_features[0].permute(1, 2, 0).reshape(-1, dense_features.shape[1])
                    seg = seg.reshape(-1)
                    sorts = seg.argsort()
                    dense_features_sort, seg_sort = dense_features[sorts], seg[sorts]
                    x = dense_features_sort
                    # The cumsum operation is the only one that takes times
                    x = x.cumsum(0)
                    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
                    elements_sumed = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
                    kept[:-1] = seg_sort[1:] != seg_sort[:-1]
                    x = x[kept]
                    x = torch.cat((x[:1], x[1:] - x[:-1]))

                    elements_sumed = elements_sumed[kept]
                    elements_sumed[1:] = elements_sumed[1:] - elements_sumed[:-1]
                    x /= elements_sumed[:, None]
                    return x
                else:
                    sparse_features = []
                    for i in range(seg.max() + 1):
                        m = seg == i
                        x, y = torch.where(m)
                        feat = dense_features[0, :, x, y].mean(dim=1)
                        sparse_features.append(feat)
                    return torch.stack(sparse_features, dim=1).T
        else:
            return dense_features
