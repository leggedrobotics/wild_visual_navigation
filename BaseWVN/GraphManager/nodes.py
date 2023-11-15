from liegroups.torch import SE3, SO3
from torch_geometric.data import Data
import os
import torch
import torch.nn.functional as F
from typing import Optional,Union,Dict
from BaseWVN.utils import ImageProjector


class BaseNode:
    """Base node data structure"""

    _name = "base_node"

    def __init__(self, timestamp: float = 0.0, pose_base_in_world: torch.tensor = torch.eye(4)):
        assert isinstance(pose_base_in_world, torch.Tensor)

        self._timestamp = timestamp
        self._pose_base_in_world = pose_base_in_world

    def __str__(self):
        return f"{self._name}_{self._timestamp}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self._name == other.name
            and self._timestamp == other.timestamp
            and torch.equal(self._pose_base_in_world, other.pose_base_in_world)
        )

    def __lt__(self, other):
        return self._timestamp < other.timestamp

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._pose_base_in_world = self._pose_base_in_world.to(device)
        
    @classmethod
    def from_node(cls, instance):
        return cls(timestamp=instance.timestamp, pose_base_in_world=instance.pose_base_in_world)

    def is_valid(self):
        return True

    def pose_between(self, other):
        """Computes pose difference (SE(3)) between this state and other

        Args:
            other (BaseNode): Other state

        Returns:
            tensor (torch.tensor): Pose difference expressed in this' frame
        """
        return other.pose_base_in_world.inverse() @ self.pose_base_in_world

    def distance_to(self, other):
        """Computes the relative distance between states

        Args:
            other (BaseNode): Other state

        Returns:
            distance (float): absolute distance between the states
        """
        # Compute pose difference, then log() to get a vector, then extract position coordinates, finally get norm
        return (
            SE3.from_matrix(self.pose_base_in_world.inverse() @ other.pose_base_in_world, normalize=True)
            .log()[:3]
            .norm()
        )

    @property
    def name(self):
        return self._name
    
    @property
    def pose_base_in_world(self):
        return self._pose_base_in_world

    @property
    def timestamp(self):
        return self._timestamp

    @pose_base_in_world.setter
    def pose_base_in_world(self, pose_base_in_world: torch.tensor):
        self._pose_base_in_world = pose_base_in_world

    @timestamp.setter
    def timestamp(self, timestamp: float):
        self._timestamp = timestamp

class MainNode(BaseNode):
    """Main node stores the minimum information required for visual decoder.
    All the information is stored on the image plane
    
    image shape (B,C,H,W):transformed image
    feat shape (B,num_segs or H*W,C): Sparse features tensor
    seg (H,W): Segmentation map
    """

    _name = "main_node"
    
    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_cam_in_world: torch.tensor = None,
        image: torch.tensor = None,
        image_projector: ImageProjector = None,
        features: Union[torch.tensor,dict] = None,
        feature_type: str = None,
        segments: torch.tensor = None,
        camera_name="cam",
        use_for_training=True,
        phy_dim:int=2,
    ):
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)
        # Initialize members
        self._pose_cam_in_world = pose_cam_in_world
        self._pose_base_in_world = pose_base_in_world
        self._image = image
        self._image_projector = image_projector
        self._camera_name = camera_name
        self._use_for_training = use_for_training
        self._features = features
        self._feature_segments = segments
        self._feature_type = feature_type
        self._is_feat_compressed=True if isinstance(features,dict) else False
        self._phy_dim=phy_dim
        """ 
        Warning: to save GPU memory, move img and features to cpu
        """
        self._image = self._image.cpu()
        if self._is_feat_compressed:
            for key, tensor in self._features.items():
                self._features[key] = tensor.cpu()
        else:
            self._features = self._features.cpu()

        
        # Uninitialized members
        self._confidence = None
        self._prediction = None
        self._supervision_mask = None
        self._supervision_signal_valid = None
    
    @property
    def camera_name(self):
        return self._camera_name

    @property
    def confidence(self):
        return self._confidence

    @property
    def feature_type(self):
        return self._feature_type
    
    @property
    def is_feat_compressed(self):
        return self._is_feat_compressed
    
    @property
    def image(self):
        return self._image
    @property
    def features(self):
        return self._features
    @property
    def feature_segments(self):
        return self._feature_segments
    @property
    def image_projector(self):
        return self._image_projector

    @property
    def pose_cam_in_world(self):
        return self._pose_cam_in_world

    @property
    def prediction(self):
        return self._prediction

    @property
    def supervision_signal_valid(self):
        return self._supervision_signal_valid

    @property
    def supervision_mask(self):
        return self._supervision_mask

    @property
    def use_for_training(self):
        return self._use_for_training

    @camera_name.setter
    def camera_name(self, camera_name):
        self._camera_name = camera_name

    @confidence.setter
    def confidence(self, confidence):
        self._confidence = confidence

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type

    @image_projector.setter
    def image_projector(self, image_projector):
        self._image_projector = image_projector

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = prediction

    @supervision_signal_valid.setter
    def supervision_signal_valid(self, _supervision_signal_valid):
        self._supervision_signal_valid = _supervision_signal_valid

    @supervision_mask.setter
    def supervision_mask(self, supervision_mask):
        self._supervision_mask = supervision_mask

    @use_for_training.setter
    def use_for_training(self, use_for_training):
        self._use_for_training = use_for_training
    
    def clear_debug_data(self):
        """Removes all data not required for training"""
        try:
            del self._image
            del self._supervision_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            pass  # Image already removed

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._image_projector.change_device(device)
        self._image = self._image.to(device)
        self._pose_base_in_world = self._pose_base_in_world.to(device)
        self._pose_cam_in_world = self._pose_cam_in_world.to(device)

        if self._features is not None and not self._is_feat_compressed:
            self._features = self._features.to(device)
        if self._feature_segments is not None:
            self._feature_segments = self._feature_segments.to(device)
        if self._prediction is not None:
            self._prediction = self._prediction.to(device)
        if self._confidence is not None:
            self._confidence = self._confidence.to(device)
        if self._supervision_mask is not None:
            self._supervision_mask = self._supervision_mask.to(device)
        if self._supervision_signal_valid is not None:
            self._supervision_signal_valid = self._supervision_signal_valid.to(device)
     
    # def is_valid(self):
    #     valid_members = (
    #         isinstance(self._features, torch.Tensor)
    #         and isinstance(self._supervision_signal, torch.Tensor)
    #         and isinstance(self._supervision_signal_valid, torch.Tensor)
    #     )
    #     valid_signals = self._supervision_signal_valid.any() if valid_members else False

    #     return valid_members and valid_signals
    
    def update_supervision_signal(self):
        if not self.is_feat_compressed:
            # it average the supervision signal over each segment in case of muliple possible value in a segment
            if self._supervision_mask is None:
                return
            signal=self._supervision_mask
            if len(signal.shape) != 3 or signal.shape[0] != self._phy_dim:
                raise ValueError("Supervision signal must be a 2 channel image (2,H,W)")
            # If we don't have features, return
            if self._features is None:
                return

            # If we have features, update supervision signal
            C,N, M = signal.shape
            num_segments = self._feature_segments.max() + 1
            # torch.arange(0, num_segments)[None, None]

            # Create array to mask by index (used to select the segments)
            multichannel_index_mask = torch.arange(0, num_segments, device=self._feature_segments.device)[
                None, None,None
            ].expand(C,N, M, num_segments)
            # Make a copy of the segments with the dimensionality of the segments, so we can split them on each channel
            multichannel_segments = self._feature_segments[None,:, :, None].expand(C,N, M, num_segments)

            # Create a multichannel mask that allows to associate a segment to each channel
            multichannel_segments_mask = multichannel_index_mask == multichannel_segments

            # Apply the mask to an expanded supervision signal and get the mean value per segment
            # First we get the number of elements per segment (stored on each channel)
            num_elements_per_segment = (
                multichannel_segments_mask * ~torch.isnan(signal[:, :, :,None].expand(C,N, M, num_segments))
            ).sum(dim=[1, 2])
            # We get the sum of all the values of the supervision signal that fall in the segment
            signal_sum = (signal.nan_to_num(0)[:, :,:, None].expand(C,N, M, num_segments) * multichannel_segments_mask).sum(
                dim=[1, 2]
            )
            # Compute the average of the supervision signal dividing by the number of elements
            signal_mean = signal_sum / num_elements_per_segment

            # self._supervision_signal = signal_mean.nan_to_num(0)
        # the not nan pixel is valid
        self._supervision_signal_valid = ~torch.isnan(self._supervision_mask)

    def recover_feat(self):
        """Recover the feature from compression if needed"""
        if not self._is_feat_compressed:
            return self._features
        else:
            # check it must be dict
            if not isinstance(self._features,dict):
                raise TypeError("The feature must be dict")
            # feat in dict: (ratio_h,ratio_w):(B, C, H, W)
            # recover to tensor (B, H*W,C)
            recover_feat=[]
            for key,value in self._features:
                scale_h,scale_w=key
                resized_feats = F.interpolate(
                    value.type(torch.float32), scale_factor=(scale_h, scale_w)
                )
                recover_feat.append(resized_feats)
            recover_feat=torch.cat(recover_feat,dim=1)
            recover_feat=recover_feat.permute(0,2,3,1)
            res_features = recover_feat.reshape(recover_feat.shape[0],recover_feat.shape[1]*recover_feat.shape[2],-1)  
            return res_features
    
    def query_valid_batch(self):
        """ Since the feat is compressed with ratio info in dict, we need to query the valid feat
            given supervision_signal_valid ---> get the dataset for model training
            
            Return:
                valid_feats: (num_valid, C)
                valid_masks: (num_valid, C (=2))
        """
        # first check the real ratio is equal to the ratio in dict
        if not self._is_feat_compressed:
            return self._features
        else:
            _,H_s,W_s=self._supervision_signal_valid.shape
            valid_feats = []
            valid_masks=[]
            for key, value in self._features.items():
                B,C,H,W=value.shape
                if H_s/H!=key[0] or W_s/W!=key[1]:
                    raise ValueError("The ratio in feats dict is not equal to the real ratio")

                valid_indices = torch.where(self._supervision_signal_valid[0] == 1)
                ratio_h,ratio_w=key
                for h_idx, w_idx in zip(*valid_indices):
                    # Extract the mask
                    mask_vec=self._supervision_mask[:,h_idx,w_idx]
                    valid_masks.append(mask_vec)
                    
                    patch_h_idx = h_idx // ratio_h
                    patch_w_idx = w_idx // ratio_w
                    # Extract the feature vector for the corresponding patch
                    feat_vec = value[:, :, patch_h_idx, patch_w_idx]
                    valid_feats.append(feat_vec)
            return torch.cat(valid_feats,dim=0),torch.stack(valid_masks,dim=0)


class SubNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    _name = "sub_node"
    def __init__(self, 
                 timestamp: float = 0, 
                 pose_base_in_world: torch.tensor = torch.eye(4),
                 phy_pred: torch.tensor = None,
                 feet_planes: torch.tensor = None,
                 feet_contact: torch.tensor = None,):
        assert isinstance(pose_base_in_world, torch.Tensor)
        assert len(feet_contact.shape)==1
        super().__init__(timestamp, pose_base_in_world)
        
        self._pose_base_in_world = pose_base_in_world
        self._phy_pred = phy_pred
        self._feet_planes = feet_planes
        self._feet_contact = feet_contact
        
    def change_device(self, device):
        super().change_device(device)
        self._pose_base_in_world = self._pose_base_in_world.to(device)
        self._phy_pred = self._phy_pred.to(device)
        self._feet_planes = self._feet_planes.to(device)
        self._feet_contact = self._feet_contact.to(device)
    
    def is_valid(self):
        return isinstance(self._phy_pred, torch.Tensor) and isinstance(self._feet_planes, torch.Tensor)
    
    @property
    def phy_pred(self):
        return self._phy_pred
    
    @property
    def feet_planes(self):
        return self._feet_planes
    
    @property
    def feet_contact(self):
        return self._feet_contact
    
    @phy_pred.setter
    def phy_pred(self, phy_pred):
        self._phy_pred = phy_pred
        
    @feet_planes.setter
    def feet_planes(self, feet_planes):
        self._feet_planes = feet_planes
        
    @feet_contact.setter
    def feet_contact(self, feet_contact):
        self._feet_contact = feet_contact
    

if __name__ == '__main__':
    H=4
    W=6
    first_tensor = torch.rand((1, H, W))
    second_tensor = torch.rand((1, H, W)) * 9 + 1  # Scaling and shifting to get values from 1 to 10
    supervision_tensor = torch.cat((first_tensor, second_tensor), dim=0)
   
    supervision_tensor[:,2:,2:]=supervision_tensor[:,2:,2:]*torch.nan

    # Generate a (3, H, W) random tensor with values from 0 to 1
    img_tensor = torch.rand((3, H, W))

    # Generate a (6, 4) tensor with values from 0 to 24
    mask_tensor = torch.arange(24).reshape(H, W)
    
    features={
    (2, 2): torch.rand(1, 10, H//2, W//2)  # Example feature tensor
    }
    
    main_node=MainNode(features=features,segments=mask_tensor,image=img_tensor)
    main_node.supervision_mask=supervision_tensor
    main_node.update_supervision_signal()
    valid_feats,valid_masks=main_node.query_valid_batch()
    # if torch.allclose(main_node.supervision_signal.reshape(2,H,W),supervision_tensor.nan_to_num(0)) :
    #     print("True")
    
    
    

    
    
    pass

