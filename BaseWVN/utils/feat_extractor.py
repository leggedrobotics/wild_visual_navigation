import torch
import torch.nn.functional as F
from .dinov2_interface import Dinov2Interface
from .focal_interface import FocalInterface
from .visualizer import plot_overlay_image,plot_tsne,add_color_bar_and_save,plot_image
from ..config import save_to_yaml
import PIL.Image
from .loss import PhyLoss
import numpy as np
from torchvision import transforms as T
from typing import Union, Dict
from BaseWVN import WVN_ROOT_DIR
from sklearn.mixture import GaussianMixture
import os
from sklearn.manifold import TSNE
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
        self.center_crop=kwargs.get('center_crop', (False,910,910))
        self.new_height=None
        self.new_width=None
        # extract crop info
        self.crop_size=self.center_crop[1:]
        self.center_crop=self.center_crop[0]
        
        if self.center_crop:
            self.target_height=self.crop_size[0]
        else:
            self.target_height=self._input_size

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
        elif self._feature_type == "focal":
            self.patch_size = 32
            self.extractor=FocalInterface(device, **kwargs)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")
        
        assert self._input_size % self.patch_size == 0, "Input size must be a multiple of patch_size"
        
        # create transform
        self.transform=self._create_transform()
        self.init_transform=False

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
        if img.shape[-2]!=self.target_height:
            transformed_img=self.transform(img)
        else:
            transformed_img=img
        # Compute segmentation
        # seg = self.compute_segments(transformed_img, **kwargs)
        # Compute features
        compressed_feats = self.compute_features(transformed_img, **kwargs)
        # Sparsify features
        # sparse_features,compressed_feats = self.sparsify_features(dense_features, seg)
        torch.cuda.empty_cache()
        return None, None,transformed_img,compressed_feats

    
    def set_original_size(self, original_width: int, original_height: int):
        if self.init_transform==False and original_height!=self.target_height:
            self.original_height = original_height
            self.original_width = original_width
            
            # somtimes the input image is already processed but we still want a smaller one for a new feat_extractor
            # so we need to recompute the transform and not scaling the processed image back to large
            if self.original_height<self._input_size:
                print("The input image is in height of {}, smaller than the planned (scale to) input H size {}, changed to the image size!".format(self.original_height,self._input_size))
                self._input_size=self.original_height
            
            self.transform=self._create_transform()
            self.init_transform=True
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
    def crop_offset(self):
        return self.crop_offset_x, self.crop_offset_y
    
    @property
    def new_size(self):
        return self.new_width, self.new_height

    def _create_transform(self):
        # Calculate aspect ratio preserving size
        aspect_ratio = self.original_width / self.original_height
        if aspect_ratio >= 1:  # If width > height, scale by height (1280 -> 448)
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
            T.Resize((new_height, new_width),self.interp,antialias=None),
            T.CenterCrop(self.crop_size) if self.center_crop else T.CenterCrop((new_height, new_width)),
            T.ConvertImageDtype(torch.float),
        ])
        
        # actual resize ratio along x and y of resize step
        self.resize_ratio_x = float(new_width) / float(self.original_width)
        self.resize_ratio_y = float(new_height) / float(self.original_height)
        if not self.center_crop:
            self.new_height=new_height
            self.new_width=new_width    
            self.crop_offset_x=0
            self.crop_offset_y=0
        else:
            self.new_height=self.crop_size[0]
            self.new_width=self.crop_size[1]
            self.crop_offset_x = (new_width - self.crop_size[1]) / 2
            self.crop_offset_y= (new_height - self.crop_size[0]) / 2
            
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
        # seg = seg.unsqueeze(0).repeat(B, 1, 1)
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
        B,C,H,W=img_internal.shape
        
        if self._feature_type=="dinov2":
            feat_dict={}
            feat=self.extractor.inference(img_internal)
            # compute ratio of original image size to feature map size
            ratio_h = H / feat.shape[-2]
            ratio_w = W / feat.shape[-1]
            feat_dict[(ratio_h,ratio_w)]=feat
        elif self._feature_type=="focal":
            feat_dict=self.extractor.inference(img_internal)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")
        
        return feat_dict
    
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

def compute_phy_mask(img:torch.Tensor,
                     feat_extractor:FeatureExtractor=None,
                     model=None,
                     loss_fn:PhyLoss=None,
                     confidence_threshold=0.8,
                     mode="fixed",
                     plot_and_save:bool=False,
                     step:int=0,
                     **kwargs):
    """ process the original_img and return the phy_mask in resized img shape(non-confident--> nan) """
    """ Shape of phy_mask: (2,H,W) H,W is the size of resized img
        Return: conf_mask (1,1,H,W) H,W is the size of resized img
    
    """

    if mode !="fixed":
        preserved=confidence_threshold
        confidence_threshold=None
    # in online mode, no need to use extract again, just input these outputs
    trans_img=kwargs.get("trans_img",None)
    compressed_feats=kwargs.get("compressed_feats",None)
    use_conf_mask=kwargs.get("use_conf_mask",True)
    if trans_img is None or compressed_feats is None:
        if feat_extractor is None:
            raise ValueError("feat_extractor is None!")
        _,_,trans_img,compressed_feats=feat_extractor.extract(img)
    feat_input,H,W=concat_feat_dict(compressed_feats)
    feat_input=feat_input.squeeze(0)
    output=model(feat_input)
    confidence,loss_reco,loss_reco_raw=loss_fn.compute_confidence_only(output,feat_input)
    confidence=confidence.reshape(H,W)
    loss_reco=loss_reco.reshape(H,W)
    
    if isinstance(output,tuple):
        phy_dim=output[1].shape[1]-output[0].shape[1]
        output=output[1]
    else:
        phy_dim=output.shape[1]-feat_input.shape[1]
    output_phy=output[:,-phy_dim:].reshape(H,W,2).permute(2,0,1)
    if use_conf_mask:
        if confidence_threshold is not None:
            # use fixed confidence threshold to segment the phy_mask
            unconf_mask=confidence<confidence_threshold
        else:
            if mode=="gmm_1d":
                # use 1d GMM with k=2 to segment the phy_mask
                # Flatten the loss_reco to fit the GMM
                loss_reco_flat = loss_reco.flatten().detach().cpu().numpy().reshape(-1, 1)
                
                # fit a 1D GMM with k=1
                gmm_k1 = GaussianMixture(n_components=1, random_state=0)
                gmm_k1.fit(loss_reco_flat)
                
                # Fit a 1D GMM with k=2
                gmm_k2 = GaussianMixture(n_components=2, random_state=0)
                gmm_k2.fit(loss_reco_flat)
                
                # Compare the losses
                if abs(gmm_k1.lower_bound_ - gmm_k2.lower_bound_)<0.1:
                    # single gaussian distribution, use the confidence threshold
                    # std factor need tuning
                    unconf_mask=confidence<preserved
                    unconf_mask=unconf_mask.to(img.device)
                else:
                    # Predict the clusters for each data point (loss value)
                    gmm_labels = gmm_k2.predict(loss_reco_flat)
                    # Assume the cluster with the larger mean loss is the unconfident one
                    unconfident_cluster = gmm_k2.means_.argmax()
                    
                    # Create a mask from GMM predictions
                    unconf_mask = (gmm_labels == unconfident_cluster).reshape(H, W)
                    unconf_mask=torch.from_numpy(unconf_mask).to(img.device)
            elif mode=="gmm_all":
                data=loss_reco_raw.detach().cpu().numpy()
                data_reduced = data[:,:10]
                gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
                gmm.fit(data_reduced)
                # mean_losses = gmm.means_.mean(axis=1)

                # # Determine which cluster has the larger mean across all dimensions
                # unconfident_cluster = mean_losses.argmax()
                # Calculate the mean of squared values for each component's mean
                mean_losses_squared = np.square(gmm.means_).mean(axis=1)

                # Determine the unconfident cluster based on higher mean squared losses
                unconfident_cluster = mean_losses_squared.argmax()

                # Predict the cluster for each sample
                gmm_labels = gmm.predict(data_reduced)

                # Create a mask where the unconfident cluster is True
                unconf_mask = (gmm_labels == unconfident_cluster).reshape(H, W)
                unconf_mask=torch.from_numpy(unconf_mask).to(img.device)
                
                pass
    else:
        # in the case we don't use confidence mask, we just set all the mask to be False
        unconf_mask=torch.zeros_like(confidence).to(img.device).type(torch.bool)
    
    mask = unconf_mask.unsqueeze(0).repeat(output_phy.shape[0], 1, 1)
    output_phy[mask] = torch.nan
    if output_phy.shape[-2]!=trans_img.shape[-2] or output_phy.shape[-1]!=trans_img.shape[-1]:
        # upsample the output
        output_phy=F.interpolate(
            output_phy.unsqueeze(0).type(torch.float32), size=trans_img.shape[-2:]
        ).squeeze(0)
    conf_mask=~unconf_mask
    conf_mask_resized=F.interpolate(conf_mask.type(torch.float32).unsqueeze(0).unsqueeze(0),size=trans_img.shape[-2:])>0
    loss_reco_resized=F.interpolate(loss_reco.unsqueeze(0).unsqueeze(0),size=trans_img.shape[-2:])
    torch.cuda.empty_cache()
    if plot_and_save:
        time=kwargs.get("time","online")
        param=kwargs.get("param",None)
        image_name = kwargs.get("image_name", "anonymous")
        
        output_dir=os.path.join(WVN_ROOT_DIR,param.offline.ckpt_parent_folder,time)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        channel_num=output_phy.shape[0]
        # process trans_img for plotting
        trans_img_uint=plot_image(trans_img.squeeze(0))
        trans_img_pil=PIL.Image.fromarray(trans_img_uint)
        trans_img_pil=rot_or_not(trans_img_pil,param)
        # process possible label_mask for plotting if given
        label_mask=kwargs.get("label_mask",None)
            
        for i in range(channel_num):
            output_phy=output_phy.detach()
            overlay_img=plot_overlay_image(trans_img, overlay_mask=output_phy, channel=i,alpha=0.7)
            out_image = PIL.Image.fromarray(overlay_img)
            rotated_image=rot_or_not(out_image,param)
        
            # process possible label_mask for plotting if given
            if label_mask is not None:
                label=label_mask.detach()
                overlay_label=plot_overlay_image(trans_img, overlay_mask=label, channel=i,alpha=0.9)
                overlay_label_img = PIL.Image.fromarray(overlay_label)
                overlay_label_img=rot_or_not(overlay_label_img,param)
            
            # Construct a filename
            if i == 0:
                filename = f"{image_name}_fric_den_pred_step_{step}_{mode}.jpg"
            elif i==1:
                filename = f"{image_name}_stiff_den_pred_step_{step}_{mode}.jpg"
            file_path = os.path.join(output_dir, filename)
            # Save the image
            # rotated_image.save(file_path)
            
            vis_imgs=[trans_img_pil]
            if label_mask is not None:
                vis_imgs.append(overlay_label_img)
            vis_imgs.append(rotated_image)
            # add colorbar to overlay image and then save
            add_color_bar_and_save(vis_imgs,i, file_path)
            
        if param is not None:
            param_path=os.path.join(output_dir,"param.yaml")
            save_to_yaml(param,param_path)
        
    # return output_phy,trans_img,confidence,conf_mask_resized
    torch.cuda.empty_cache()
    return {"output_phy":output_phy,
            "trans_img":trans_img,
            "confidence":confidence,
            "conf_mask":conf_mask_resized,
            "loss_reco":loss_reco_resized,
            "loss_reco_raw":loss_reco_raw,
            "conf_mask_raw":conf_mask,}

def rot_or_not(img,param):
    if param is not None:
        if isinstance(img,PIL.Image.Image) and "v4l2" in param.roscfg.camera_topic:
            # Rotate the image by 180 degrees
            img = img.rotate(180)
    return img

def compute_pred_phy_loss(img:torch.Tensor,
                          conf_mask:torch.Tensor,
                          ori_phy_mask:torch.Tensor,
                          pred_phy_mask:torch.Tensor, 
                          **kwargs):
    """ 
    To calculate the mean error of predicted physical params value in confident area of a image
    conf_mask (1,1,H,W) H,W is the size of resized img
    phy_mask (2,H,W) H,W is the size of resized img
    """
    # check dim of phy_masks first
    if ori_phy_mask.shape[-2]!=pred_phy_mask.shape[-2] or ori_phy_mask.shape[-1]!=pred_phy_mask.shape[-1]:
        raise ValueError("ori_phy_mask and pred_phy_mask should have the same shape!")
    compare_regions=~torch.isnan(ori_phy_mask)
    regions_in_pred=pred_phy_mask*compare_regions
    regions_in_pred=torch.where(regions_in_pred==0,torch.nan,regions_in_pred)
    delta=torch.abs(regions_in_pred-ori_phy_mask)
    delta_mask=~torch.isnan(delta[0])
    # parent=torch.sum(delta_mask)
    # fric_mean_deviat=torch.nansum(delta[0])/parent
    # stiff_mean_deviat=torch.nansum(delta[1])/parent
    
    fric_dvalues=delta[0][delta_mask]
    fric_mean_deviation=torch.mean(fric_dvalues)
    fric_std_deviation = torch.std(fric_dvalues)
    
    stiff_dvalues=delta[1][delta_mask]
    stiff_mean_deviation=torch.mean(stiff_dvalues)
    stiff_std_deviation = torch.std(stiff_dvalues)
    
    return {"fric_mean_deviat":fric_mean_deviation,
            "fric_std_deviation":fric_std_deviation,
            "fric_loss_raw":fric_dvalues,
            "stiff_mean_deviat":stiff_mean_deviation,
            "stiff_std_deviation":stiff_std_deviation,
            "stiff_loss_raw":stiff_dvalues,
            }
    
    
    
    

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
   