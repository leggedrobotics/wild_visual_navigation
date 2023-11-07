import torch
import torch.nn.functional as F
from dinov2_interface import Dinov2Interface
import numpy as np
from torchvision import transforms as T
from typing import Union, Dict
from BaseWVN import WVN_ROOT_DIR

class FeatureExtractor:
    def __init__(
        self, device: str, segmentation_type: str = "pixel", feature_type: str = "dinov2", input_size: int = 448, **kwargs
    ):
        """Feature extraction from image

        Args:
            device (str): Compute device

        """
        self._device = device
        self._segmentation_type = segmentation_type
        self._feature_type = feature_type
        self._input_size = input_size
        # input size must be a multiple of 14
        self.patch_size = 14
        assert self._input_size % self.patch_size == 0, "Input size must be a multiple of patch_size"
        # Extract original_width and original_height from kwargs if present
        self.original_width = kwargs.get('original_width', 1920)  # Default to 1920 if not provided
        self.original_height = kwargs.get('original_height', 1280)  # Default to 1080 if not provided
        self._input_interp=kwargs.get('interp', 'bilinear')
        self.center_crop=kwargs.get('center_crop', False)

        # Interpolation type
        if self._input_interp == "bilinear":
            self.interp = T.InterpolationMode.BILINEAR
        elif self._input_interp == "nearest":
            self.interp = T.InterpolationMode.NEAREST
        elif self._input_interp == "bicubic":
            self.interp = T.InterpolationMode.BICUBIC

        self.transform=self._create_transform()

        # feature extractor
        if self._feature_type == "dinov2":
            self.extractor=Dinov2Interface(device, **kwargs)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")
        
        # segmentation
        if self._segmentation_type == "pixel":
            pass
        elif self._segmentation_type == "slic":
            from fast_slic import Slic
            self.slic = Slic(
                num_components=kwargs.get("slic_num_components", 200), compactness=kwargs.get("slic_compactness", 10)
            )
        else:
            raise ValueError(f"Segmentation[{self._segmentation_type}] not supported!")
    
    def extract(self, img, **kwargs):
        """Extract features from image

        Args:
            img (torch.tensor): Image tensor

        Returns:
            sparse_features (torch.tensor, shape:(num_segs or H*W,C)): Sparse features tensor
            seg (torch.tensor, shape:(H,W)): Segmentation map
        """
        img=self.transform(img)
        # Compute segmentation
        seg = self.compute_segments(img, **kwargs)
        # Compute features
        dense_features = self.compute_features(img, **kwargs)
        # Sparsify features
        sparse_features = self.sparsify_features(dense_features, seg)
        return sparse_features, seg

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def segmentation_type(self):
        return self._segmentation_type

    def _create_transform(self):
        # Calculate aspect ratio preserving size
        aspect_ratio = self.original_width / self.original_height
        if aspect_ratio > 1:  # If width > height, scale by height (1280 -> 448)
            new_height = self._input_size
            new_width = int(new_height * aspect_ratio)
            # check if new_width is a multiple of self.patch_size
            if new_width % self.patch_size != 0:
                new_width=new_width-new_width%self.patch_size
        else:  # If height >= width, scale by width (1920 -> 448)
            new_width = self._input_size
            new_height = int(new_width / aspect_ratio)
            # check if new_height is a multiple of self.patch_size
            if new_height % self.patch_size != 0:
                new_height=new_height-new_height%self.patch_size

        # Resize and then center crop to the expected input size
        transform = T.Compose([
            T.Resize((new_height, new_width),self.interp),
            T.CenterCrop(self._input_size) if self.center_crop else T.CenterCrop((new_height, new_width)),
            T.ConvertImageDtype(torch.float),
        ])
        self.new_height=new_height
        self.new_width=new_width
        return transform

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._device = device
        self.extractor.change_device(device)

    def compute_segments(self, img: torch.tensor, **kwargs):
        if self._segmentation_type=="pixel":
            seg=self.segment_pixelwise(img, **kwargs)
        elif self._segmentation_type=="slic":
            seg=self.segment_slic(img, **kwargs)
        return seg
    
    def segment_pixelwise(self, img: torch.tensor, **kwargs):
        # Generate pixel-wise segmentation
        B, C, H, W = img.shape
        seg = torch.arange(0, H * W, 1).reshape(H, W).to(self._device)
        return seg
    
    def segment_slic(self, img: torch.tensor, **kwargs):
        # transform image to numpy
        B, C, H, W = img.shape
        img_np = img.cpu().numpy()
        seg = self.slic.iterate(np.uint8(np.ascontiguousarray(img_np) * 255))
        return torch.from_numpy(seg).to(self._device).type(torch.long)
    
    def compute_features(self, img: torch.tensor, **kwargs):
        img_internal=img.clone()
        if self._feature_type=="dinov2":
            feat=self.extractor.inference(img_internal)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")
        return feat
    
    def sparsify_features(self, dense_features: Union[torch.Tensor, Dict[str, torch.Tensor]], seg: torch.tensor, cumsum_trick=False):
        """ Sparsify features
         
        return (H*W/segs_num,C)
          
            """
        if isinstance(dense_features, dict):
            # Multiscale feature pyramid extraction
            scales_x = [  seg.shape[0]/feat.shape[2] for feat in dense_features.values()]
            scales_y = [ seg.shape[1] /feat.shape[3] for feat in dense_features.values()]
            # upsampling the feat of each scale
            resized_feats = [
                F.interpolate(
                    feat.type(torch.float32), scale_factor=(scale_x, scale_y)
                )
                for scale_x, scale_y,feat in zip(scales_x, scales_y,dense_features.values())
            ]
            sparse_features = []

            for i in range(seg.max() + 1):
                single_segment_feature = []
                for resized_feat in resized_feats:
                    m=seg==i
                    x, y = torch.where(m)
                    avg_feat_per_seg = torch.mean(resized_feat[:, :, x, y], dim=-1)
                    single_segment_feature.append(avg_feat_per_seg)
                single_segment_feature = torch.cat(single_segment_feature, dim=1)
                sparse_features.append(single_segment_feature)
            
        else:
            # check if las two dim of seg is equal to dense_features dim, if not ,resize the feat
            if seg.shape[-2:] != dense_features.shape[-2:]:
                dense_features = F.interpolate(
                    dense_features.type(torch.float32), size=seg.shape[-2:]
                )
            # Sparsify features
            sparse_features = []
            if self._segmentation_type != "pixel":
                for i in range(seg.max() + 1):
                    m = seg == i
                    x, y = torch.where(m)
                    avg_feat_per_seg = torch.mean(dense_features[:, :, x, y], dim=-1)
                    sparse_features.append(avg_feat_per_seg)
            else:
                dense_features=dense_features.permute(0,2,3,1)
                sparse_features = dense_features.reshape(dense_features.shape[0],dense_features.shape[1]*dense_features.shape[2],-1)
                return sparse_features
        # Concatenate features
        
        sparse_features = torch.stack(sparse_features, dim=1)
        return sparse_features

def test_extractor():
    import cv2
    import os
    image_relative_path = 'image/hiking.png'  # Update to the relative path of your image
    feat_relative_path = 'image/hiking_feat_low.png'
    # Use os.path.join to get the full path of the image
    image_path = os.path.join(WVN_ROOT_DIR, image_relative_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np_img = cv2.imread(image_path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    input_size=1260
    extractor=FeatureExtractor(device, input_size=input_size, original_width=img.shape[-1], original_height=img.shape[-2], interp='bilinear')
    features, seg=extractor.extract(img)
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    B,N,C=features.shape
    features=features[0].cpu().numpy()
    # B,C,H,W=features.shape
    # features=features.permute(0,2,3,1)
    # features=features.reshape(B,H*W,C)
    # features=features[0].cpu().numpy()
    n=3
    pca = PCA(n_components=n)
    pca.fit(features)

    pca_features = pca.transform(features)
    for i in range(n):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
    # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    feat_height = seg.shape[0]
    feat_width = seg.shape[1]
    plt.imshow(pca_features.reshape(feat_height, feat_width, n).astype(np.uint8))
    image_path = os.path.join(WVN_ROOT_DIR, feat_relative_path)
    plt.savefig(image_path)



                
if __name__=="__main__":
    import torch
    import torch.nn.functional as F

    # Original segmentation map
    seg = torch.tensor([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [2, 2, 3, 3],
    [2, 2, 3, 3]
    ])
    print(seg.reshape(-1))
    # Scale factors
    scales_x = [2, 0.5]
    scales_y = [2, 0.5]

    # Resized segmentation maps
    segs = [
        F.interpolate(
            seg[None, None, :, :].type(torch.float32), scale_factor=(scale_x, scale_y), mode='nearest'
        )[0, 0].type(torch.long)
        for scale_x, scale_y in zip(scales_x, scales_y)
    ]

    # Print the resized maps
    for i, resized_seg in enumerate(segs):
        print(f"Resized seg at scale {scales_x[i]}x{[scales_y[i]]}:")
        print(resized_seg.numpy(), "\n")
    
    dense_features = {
        "feat_1":torch.rand((8, 4, 4)).unsqueeze(0),
        "feat_2":torch.rand((16, 2, 2)).unsqueeze(0),

    }

    test_extractor()

    # if isinstance(dense_features, dict):
    #     # Multiscale feature pyramid extraction
    #     scales_x = [  seg.shape[0]/feat.shape[2] for feat in dense_features.values()]
    #     scales_y = [ seg.shape[1] /feat.shape[3] for feat in dense_features.values()]
    #     # upsampling the feat of each scale
    #     resized_feats = [
    #         F.interpolate(
    #             feat.type(torch.float32), scale_factor=(scale_x, scale_y)
    #         )
    #         for scale_x, scale_y,feat in zip(scales_x, scales_y,dense_features.values())
    #     ]
    #     sparse_features = []

    #     for i in range(seg.max() + 1):
    #         single_segment_feature = []
    #         for resized_feat in resized_feats:
    #             m=seg==i
    #             x, y = torch.where(m)
    #             avg_feat_per_seg = torch.mean(resized_feat[:, :, x, y], dim=-1)
    #             single_segment_feature.append(avg_feat_per_seg)
    #         single_segment_feature = torch.cat(single_segment_feature, dim=1)
    #         sparse_features.append(single_segment_feature)
    #     # check if feat dim is 4
    #     if len(sparse_features[0].shape)==2:
    #         sparse_features = torch.cat(sparse_features, dim=0)
    #     else:
    #         sparse_features = torch.stack(sparse_features, dim=0)
    #     pass

  