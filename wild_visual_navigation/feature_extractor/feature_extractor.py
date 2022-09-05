from wild_visual_navigation.feature_extractor import StegoInterface, DinoInterface, SegmentExtractor

import torch.nn.functional as F
import skimage
import torch
import numpy as np
import kornia
from kornia.feature import DenseSIFTDescriptor
from kornia.contrib import extract_tensor_patches, combine_tensor_patches
from torchvision import transforms as T
from PIL import Image, ImageDraw
from wild_visual_navigation.utils import Timer

class FeatureExtractor:
    def __init__(self, device: str, segmentation_type: str = "slic", feature_type: str = "dino"):
        """Feature extraction from image

        Args:
            device (str): Compute device
            extractor (str): Extractor model: stego, dino_slic
        """

        self._device = device
        self._segmentation_type = segmentation_type
        self._feature_type = feature_type

        # Prepare segment extractor
        self.segment_extractor = SegmentExtractor().to(self._device)

        # Prepare extractor depending on the type
        if self._feature_type == "stego":
            self._feature_dim = 90
            self.extractor = StegoInterface(device=device)
        elif self._feature_type == "dino":
            self._feature_dim = 90
            self.extractor = DinoInterface(device=device)
        elif self._feature_type == "sift":
            self._feature_dim = 128
            self.extractor = DenseSIFTDescriptor().to(device)
        elif self._feature_type == "histogram":
            self._feature_dim = 90
        else:
            raise f"Extractor[{self._feature_type}] not supported!"

    def extract(self, img, **kwargs):
        # Compute segments, their centers, and edges connecting them (graph structure)
        edges, seg, center = self.compute_segments(img, **kwargs)
        
        # Compute features
        dense_feat = self.compute_features(img, seg, center, **kwargs)
        assert (
            len(dense_feat.shape) == 4
        ), f"dense_feat has incorrect shape size {dense_feat.shape} (should be B, C, H, W)"
        
        # Sparsify features to match the centers if required
        feat = self.sparsify_features(dense_feat, seg)

        return edges, feat, seg, center
        # return getattr(self, self._feature_type)(img, **kwargs)

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
        if kwargs.get("cell_size", None) is not None:
            cell_size = kwargs["cell_size"]
        else:
            cell_size = 16
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
        if kwargs.get("n_segments", None) is not None:
            n_segments = kwargs["n_segments"]
        else:
            n_segments = 100

        if kwargs.get("compactness", None) is not None:
            compactness = kwargs["compactness"]
        else:
            compactness = 10.0

        # Get slic clusters
        img_np = kornia.utils.tensor_to_image(img)
        seg = skimage.segmentation.slic(
            img_np, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=2
        )
        seg = torch.from_numpy(seg).to(self._device)

        return seg

    def segment_stego(self, img, **kwargs):
        # Prepare input image
        img_internal = img.clone()
        self.extractor.inference_crf(img_internal)
        seg = torch.from_numpy(self.extractor.linear_segments).to(self._device)

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
    def compute_stego(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        return self.extractor.features

    def sparsify_features(self, dense_features: torch.tensor, seg: torch.tensor):
        if self._feature_type not in ["histogram"] and self._segmentation_type not in ["none"]:
            # Get median features for each cluster
            sparse_features = []
            for i in range(seg.max() + 1):
                m = seg == i
                x, y = torch.where(m)
                feat = dense_features[0, :, x, y].median(dim=1)[0]
                sparse_features.append(feat)
            return torch.stack(sparse_features, dim=1).T
        else:
            return dense_features
