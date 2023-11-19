import torch
import torch.nn.functional as F
from .dinov2_interface import Dinov2Interface
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

        # feature extractor
        if self._feature_type == "dinov2":
            self.patch_size = 14
            self.extractor=Dinov2Interface(device, **kwargs)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")
        
        assert self._input_size % self.patch_size == 0, "Input size must be a multiple of patch_size"
        
        # create transform
        self.transform=self._create_transform()

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
            img (torch.tensor): Image tensor (B,C,H,W)

        Returns:
            sparse_features (torch.tensor, shape:(B,num_segs or H*W,C)): Sparse features tensor
            seg (torch.tensor, shape:(H,W)): Segmentation map
            transformed_img (torch.tensor, shape:(B,C,H,W)): Transformed image
            compressed_feats (Dict): only in pixel segmentation, {(scale_h,scale_w):feat-->(B,C,H,W))}
        """
        transformed_img=self.transform(img)
        # Compute segmentation
        seg = self.compute_segments(transformed_img, **kwargs)
        # Compute features
        dense_features = self.compute_features(transformed_img, **kwargs)
        # Sparsify features
        sparse_features,compressed_feats = self.sparsify_features(dense_features, seg)
        return sparse_features, seg,transformed_img,compressed_feats

    
    def set_original_size(self, original_width: int, original_height: int):
        self.original_height = original_height
        self.original_width = original_width
        self.transform=self._create_transform()
        return self.original_width, self.original_height

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def segmentation_type(self):
        return self._segmentation_type
    
    @property
    def resize_ratio(self):
        return self.resize_ratio_x, self.resize_ratio_y
    
    @property
    def new_size(self):
        return self.new_width, self.new_height

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
        if not self.center_crop:
            self.new_height=new_height
            self.new_width=new_width    
        else:
            self.new_height=self._input_size
            self.new_width=self._input_size
        # actual resize ratio along x and y
        self.resize_ratio_x = float(self.new_width) / float(self.original_width)
        self.resize_ratio_y = float(self.new_height) / float(self.original_height)
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
        img=img.permute(0,2,3,1)
        img_np = img[0].cpu().numpy()
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
        Input:
            dense_features (B,C,H,W): dense features tensor or dict of dense features tensors
            seg (H,W): segmentation map
         
        Return:
            sparse_features (B,H*W/segs_num,C)
            compressed_feat (Dict): only in pixel segmentation
          
            """
        compressed_feat={}
        if isinstance(dense_features, dict):
            compressed_feat=None
            # Multiscale feature pyramid extraction
            scales_h = [  seg.shape[0]/feat.shape[2] for feat in dense_features.values()]
            scales_w = [ seg.shape[1] /feat.shape[3] for feat in dense_features.values()]
            # upsampling the feat of each scale
            resized_feats = [
                F.interpolate(
                    feat.type(torch.float32), scale_factor=(scale_h, scale_w)
                )
                for scale_h, scale_w,feat in zip(scales_h, scales_w,dense_features.values())
            ]
            sparse_features = []
            if self._segmentation_type != "pixel":
                compressed_feat=None
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
                for scale_h, scale_w,feat in zip(scales_h, scales_w,dense_features.values()):
                    compressed_feat[(scale_h,scale_w)]=feat
                resized_feats=torch.cat(resized_feats,dim=1)
                resized_feats=resized_feats.permute(0,2,3,1)
                sparse_features = resized_feats.reshape(resized_feats.shape[0],resized_feats.shape[1]*resized_feats.shape[2],-1)
                return sparse_features,compressed_feat
            
        else:
            # check if las two dim of seg is equal to dense_features dim, if not ,resize the feat
            scale_h=seg.shape[0]/dense_features.shape[2]
            scale_w=seg.shape[1]/dense_features.shape[3]
            if seg.shape[-2:] != dense_features.shape[-2:]:
                resized_features = F.interpolate(
                    dense_features.type(torch.float32), scale_factor=(scale_h, scale_w)
                )
            # Sparsify features
            sparse_features = []
            if self._segmentation_type != "pixel":
                compressed_feat=None
                for i in range(seg.max() + 1):
                    m = seg == i
                    x, y = torch.where(m)
                    avg_feat_per_seg = torch.mean(resized_features[:, :, x, y], dim=-1)
                    sparse_features.append(avg_feat_per_seg)
            else:
                compressed_feat[(scale_h,scale_w)]=dense_features
                resized_features=resized_features.permute(0,2,3,1)
                sparse_features = resized_features.reshape(resized_features.shape[0],resized_features.shape[1]*resized_features.shape[2],-1)
                return sparse_features,compressed_feat
        
        # Concatenate features
        sparse_features = torch.stack(sparse_features, dim=1)
        return sparse_features,compressed_feat

def concat_feat_dict(feat_dict: Dict[tuple, torch.Tensor]):
    """ Concatenate features from different scales, all upsamples to the first scale (expected to be the highest resolution) """
    """ Return: sparse_features (B,H*W/segs_num,C)
        feat_height: H
        feat_width: W
    """
    first_shape = list(feat_dict.values())[0].shape
    scales_h = [  first_shape[2]/feat.shape[2] for feat in feat_dict.values()]
    scales_w = [ first_shape[3] /feat.shape[3] for feat in feat_dict.values()]
    # upsampling the feat of each scale
    resized_feats = [
        F.interpolate(
            feat.type(torch.float32), scale_factor=(scale_h, scale_w)
        )
        for scale_h, scale_w,feat in zip(scales_h, scales_w,feat_dict.values())
    ]
    resized_feats=torch.cat(resized_feats,dim=1)
    resized_feats=resized_feats.permute(0,2,3,1)
    sparse_features = resized_feats.reshape(resized_feats.shape[0],resized_feats.shape[1]*resized_feats.shape[2],-1)
    return sparse_features,first_shape[2],first_shape[3]

def test_extractor():
    import cv2
    import os
    import time
    image_relative_path = 'image/sample.png'  # Update to the relative path of your image
    feat_relative_path = 'image/sample_feat.png'
    # Use os.path.join to get the full path of the image
    image_path = os.path.join(WVN_ROOT_DIR, image_relative_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np_img = cv2.imread(image_path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    input_size=1260
    extractor=FeatureExtractor(device, segmentation_type="pixel",input_size=input_size, original_width=img.shape[-1], original_height=img.shape[-2], interp='bilinear')
    start_time = time.time()
    
    features, seg,trans_img,_=extractor.extract(img)

     # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    inference_time = end_time - start_time

    # Print the resulting feature tensor and inference time
    print("Feature shape:", features.shape)
    print("Extract time: {:.3f} seconds".format(inference_time))
    
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
    scales_h = [2, 0.5]
    scales_w = [2, 0.5]

    # Resized segmentation maps
    segs = [
        F.interpolate(
            seg[None, None, :, :].type(torch.float32), scale_factor=(scale_h, scale_w), mode='nearest'
        )[0, 0].type(torch.long)
        for scale_h, scale_w in zip(scales_h, scales_w)
    ]

    # Print the resized maps
    for i, resized_seg in enumerate(segs):
        print(f"Resized seg at scale {scales_h[i]}x{[scales_w[i]]}:")
        print(resized_seg.numpy(), "\n")
    
    dense_features = {
        "feat_1":torch.ones((1, 4, 4)).unsqueeze(0),
        "feat_2":torch.ones((2, 2, 2)).unsqueeze(0)*2,

    }
    sparse_features=concat_feat_dict(dense_features)
    test_extractor()

   
  