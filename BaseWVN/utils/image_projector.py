from BaseWVN import WVN_ROOT_DIR
from pytictac import Timer
import os
from os.path import join
import torch
from torchvision import transforms as T
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.geometry.linalg import transform_points
from kornia.utils.draw import draw_convex_polygon
from liegroups.torch import SE3, SO3

class ImageProjector:
    def __init__(self, K: torch.tensor, h: int, w: int):
        """Initializes the projector using the pinhole model, without distortion

        Args:
            K (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Camera matrices
            h (torch.Tensor, dtype=torch.int64): Image height
            w (torch.Tensor, dtype=torch.int64): Image width

        Returns:
            None
        """
        # Get device for later
        device = K.device
        
        # Initialize pinhole model (no extrinsics)
        E = torch.eye(4).expand(K.shape).to(device)
        h=torch.IntTensor([h]).to(device)
        w=torch.IntTensor([w]).to(device)
        # Store parameters
        self.K = K
        self.height = h
        self.width = w
        # Initialize camera with scaled parameters
        self.camera=PinholeCamera(K,E,h,w)
        
        # Preallocate masks
        B = self.camera.batch_size
        C = 3  # RGB channel output
        H = self.camera.height.item()
        W = self.camera.width.item()
        # Create output mask
        self.masks = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)

    @property
    def scaled_camera_matrix(self):
        return self.camera.intrinsics.clone()[:3, :3]
    
    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.K = self.K.to(device)
        self.camera = PinholeCamera(
            self.camera.intrinsics.to(device),
            self.camera.extrinsics.to(device),
            self.camera.height.to(device),
            self.camera.width.to(device),
        )
    
    def check_validity(self, points_3d: torch.tensor, points_2d: torch.tensor) -> torch.tensor:
        """Check that the points are valid after projecting them on the image

        Args:
            points_3d: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches of N points in camera frame
            points_2d: (torch.Tensor, dtype=torch.float32, shape=(B, N, 2)): B batches of N points on the image

        Returns:
            valid_points: (torch.Tensor, dtype=torch.bool, shape=(B, N, 1)): B batches of N bools
        """

        # Check cheirality (if points are behind the camera, i.e, negative z)
        valid_z = points_3d[..., 2] >= 0
        # # Check if projection is within image range
        valid_xmin = points_2d[..., 0] >= 0
        valid_xmax = points_2d[..., 0] <= self.camera.width
        valid_ymin = points_2d[..., 1] >= 0
        valid_ymax = points_2d[..., 1] <= self.camera.height

        # Return validity
        return valid_z & valid_xmax & valid_xmin & valid_ymax & valid_ymin, valid_z
    
    def project(self, pose_camera_in_world: torch.tensor, points_W: torch.tensor):
        """Applies the pinhole projection model to a batch of points

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches of N input points in world frame

        Returns:
            projected_points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 2)): B batches of N output points on image space
        """

        # Adjust input points depending on the extrinsics
        T_CW = pose_camera_in_world.inverse()
        # Convert from fixed to camera frame
        points_C = transform_points(T_CW, points_W)

        # Project points to image
        projected_points = self.camera.project(points_C)

        # Validity check (if points are out of the field of view)
        valid_points, valid_z = self.check_validity(points_C, projected_points)

        # Return projected points and validity
        return projected_points, valid_points, valid_z
    
    def project_and_render(
        self, pose_camera_in_world: torch.tensor, points: torch.tensor, colors: torch.tensor, image: torch.tensor = None
    ):
        """Projects the points and returns an image with the projection

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches, of N input points in 3D space
            colors: (torch.Tensor, rtype=torch.float32, shape=(B, 3))

        Returns:
            out_img (torch.tensor, dtype=torch.int64): Image with projected points
        """

        # self.masks = self.masks * 0.0
        B = self.camera.batch_size
        C = 3  # RGB channel output
        H = self.camera.height.item()
        W = self.camera.width.item()
        self.masks = torch.zeros((B, C, H, W), dtype=torch.float32, device=self.camera.camera_matrix.device)
        image_overlay = image

        # Project points
        projected_points, valid_points, valid_z = self.project(pose_camera_in_world, points)

        # Mask invalid points
        # projected_points[~valid_points,:] = torch.nan
        projected_points[~valid_z, :] = torch.nan
        # projected_points[projected_points < 0.0]

        # Fill the mask
        self.masks = draw_convex_polygon(self.masks, projected_points, colors)

        # Draw on image (if applies)
        if image is not None:
            if len(image.shape) != 4:
                image = image[None]
            image_overlay = draw_convex_polygon(image, projected_points, colors)

        # Return torch masks
        self.masks[self.masks == 0.0] = torch.nan

        return self.masks, image_overlay, projected_points, valid_points