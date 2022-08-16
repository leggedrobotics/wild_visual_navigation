from wild_visual_navigation.feature_extractor import StegoInterface, DinoInterface, SegmentExtractor
from wild_visual_navigation.utils import PlotHelper

import torch.nn.functional as F
import skimage
import torch
import numpy as np
import kornia
from torchvision import transforms as T
from PIL import Image, ImageDraw


class FeatureExtractor:
    def __init__(self, device: str, segmentation_type: str = "slic", extractor_type: str = "dino"):
        """Feature extraction from image

        Args:
            device (str): Compute device
            extractor (str): Extractor model: stego, dino_slic
        """

        self.device = device
        self.segmentation_type = segmentation_type
        self.extractor_type = extractor_type

        # Prepare segment extractor
        if self.segmentation_type == "slic" or self.segmentation_type == "stego":
            self.segmen_extractor = SegmentExtractor().to(self.device)

        # Prepare extractor depending on the type
        if self.extractor_type == "stego":
            self.extractor = StegoInterface(device=device)
        elif self.extractor_type == "dino":
            self.extractor = DinoInterface(device=device)
        else:
            raise f"Extractor[{self.extractor_type}] not supported!"

    def extract(self, img, **kwargs):
        # Compute segments, their centers, and edges connecting them (graph structure)
        edges, seg, center = self.compute_segments(img)

        # Compute features
        dense_features = self.compute_features(img, seg, center)

        # Sparsify features to match the centers if required
        feat = self.sparsify_features(img, seg, center)

        return edges, feat, seg, center
        # return getattr(self, self.extractor_type)(img, **kwargs)

    def get_extractor_type(self):
        return self.extractor_type

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.device = device
        self.extractor.change_device(device)

    def compute_segments(self, img: torch.tensor, **kwargs):
        if self.extractor_type == "none" or self.segmentation_type is None:
            edges, seg, center = self.segment_pixelwise(img, **kwargs)
            
        elif self.extractor_type == "grid":
            edges, seg, center = self.segment_grid(img, **kwargs)

        elif self.extractor_type == "slic":
            edges, seg, center = self.segment_slic(img, **kwargs)

        elif self.extractor_type == "stego":
            edges, seg, center = self.segment_stego(img, **kwargs)
        else:
            raise f"segmentation_type [{self.segmentation_type}] not supported"

        return edges, seg, center 

    def segment_pixelwise(self, img, **kwargs):
        pass

    def segment_grid(self, img, **kwargs):
        pass

    def segment_slic(self, img, **kwargs):
        # Get slic clusters
        img_np = kornia.utils.tensor_to_image(img)
        seg = skimage.segmentation.slic(
            img_np, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=2
        )
        seg = torch.from_numpy(seg).to(self.device)

        # Extract adjacency_list based on segments
        adjacency_list = self.segment_extractor.adjacency_list(seg[None, None])

        # Extract centers
        centers = self.segment_extractor.centers(seg[None, None])

        return edges, seg, center

    def segment_stego(self, img, **kwargs):
        # Prepare input image
        img_internal = img.clone()
        self.extractor.inference(img_internal)
        seg = self.extractor.segments

        # Extract adjacency_list based on segments
        adjacency_list = self.segment_extractor.adjacency_list(seg[None, None])

        # Extract centers
        centers = self.segment_extractor.centers(seg[None, None])

        return edges, seg, center
        
    def compute_features(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        if self.extractor_type == "histogram":
            feat = self.compute_histogram(img, seg, center, **kwargs)

        elif self.extractor_type == "sift":
            feat = self.compute_sift(img, seg, center, **kwargs)

        elif self.extractor_type == "dino":
            feat = self.compute_dino(img, seg, center, **kwargs)

        elif self.extractor_type == "stego":
            feat = self.compute_stego(img, seg, center, **kwargs)
        else:
            raise f"segmentation_type [{self.segmentation_type}] not supported"

        return feat
    
    def compute_histogram(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        pass

    def compute_sift(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        pass

    def compute_dino(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        img_internal = img.clone()
        features = self.extractor.inference(img_internal)
        return features

    def compute_stego(self, img: torch.tensor, seg: torch.tensor, center: torch.tensor, **kwargs):
        return self.extractor.features

    @torch.no_grad()
    def dino_slic(
        self,
        img: torch.tensor,
        n_segments: int = 200,
        compactness: float = 10.0,
        return_centers: bool = False,
        return_image: bool = False,
        show: bool = False,
    ):
        """Extract dino feature form image

        Args:
            img (torch.Tensor, dtype=torch.float32, shape=(BS, C, H, W)): Image
            n_segments (int, optional): Defaults to 100.
            compactness (float, optional): Defaults to 100.0.
            return_centers (bool, optional): Defaults to False.
            return_image (bool, optional): Defaults to False.
            show (bool, optional): Defaults to False.

        Returns:
            adj (torch.Tensor, dtype=torch.uint32, shape=(2,N)): Graph Structure
            feat (torch.Tensor, dtype=torch.float32, shape=(N, C)): Features
            seg (torch.Tensor, dtype=torch.uint32, shape=(H, W)): Segmentation
            center (torch.Tensor, dtype=torch.float32, shape=(2,N)): Center of custer in image plane
        """

        # Currently on BS=1 supported
        assert img.shape[0] == 1 and len(img.shape) == 4

        # Prepare input image
        img_internal = img.clone()
        # img_internal = self.crop(img_internal.to(self.device))
        img_internal = self.norm(img_internal)

        # Extract dino features
        feat_dino = self.extractor.inference(img_internal, interpolate=False)

        # Fix size of DINO features to match input image's size
        B, D, H, W = img.shape
        new_size = (H, H)
        pad = int((W - H) / 2)
        feat_dino = F.interpolate(feat_dino, new_size, mode="bilinear", align_corners=True)
        feat_dino = F.pad(feat_dino, pad=[pad, pad, 0, 0])

        # Get slic clusters
        img_np = kornia.utils.tensor_to_image(img)
        seg = skimage.segmentation.slic(
            img_np, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=2
        )

        # extract adjacency_list based on clusters
        seg = torch.from_numpy(seg).to(self.device)
        adjacency_list = self.se.adjacency_list(seg[None, None])

        # get mean dino features for each cluster
        features = []
        for i in range(seg.max() + 1):
            m = seg == i
            x, y = torch.where(m)
            feat = feat_dino[0, :, x, y].median(dim=1)[0]
            features.append(feat)

        ret = (adjacency_list.T, torch.stack(features, dim=1).T, seg)

        if return_centers:
            ret += (self.se.centers(seg[None, None]),)

        if show or return_image:
            ph = PlotHelper()
            ph.add(img_np, "Input Image Cropped")
            ph.add(seg, "SLIC Segmentation")

            img_pil = Image.fromarray(np.uint8(img_np * 255))
            seg_pil = Image.fromarray(np.uint8(seg.cpu().numpy()[0, 0]))
            img_draw = ImageDraw.Draw(img_pil)
            seg_draw = ImageDraw.Draw(seg_pil)

            centers = self.se.centers(seg)
            fil_col = (seg.max() + 5).item()
            for i in range(adjacency_list.shape[1]):
                a, b = adjacency_list[0, i], adjacency_list[1, i]
                line_params = centers[a].tolist() + centers[b].tolist()
                img_draw.line(line_params, fill=fil_col)
                seg_draw.line(line_params, fill=fil_col)

            for i in range(centers.shape[0]):
                params = centers[i].tolist()
                params = [p - 2 for p in params] + [p + 2 for p in params]
                img_draw.ellipse(params, width=10, fill=fil_col + 1)
                seg_draw.ellipse(params, width=10, fill=fil_col + 1)

            ph.add(np.array(img_pil), "Image Feature-Graph")
            ph.add(np.array(seg_pil), "SLIC Feature-Graph")

            if show:
                ph.show()

            if return_image:
                ret += (ph.get_img(),)

        return ret

    def stego(self, img: torch.tensor):
        # currently on BS=1 supported
        assert img.shape[0] == 1 and len(img.shape) == 4

        # Prepare input image
        img_internal = img.clone()
        # img_internal = self.crop(img_internal.to(self.device))
        img_internal = self.norm(img_internal)

        return self.extractor.inference(img_internal)
