#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from pytictac import Timer
from os.path import join
import torch
from torchvision import transforms as T
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.geometry.linalg import transform_points
from kornia.utils.draw import draw_convex_polygon
from liegroups.torch import SE3, SO3


class ImageProjector:
    def __init__(self, K: torch.tensor, h: int, w: int, new_h: int = None, new_w: int = None):
        """Initializes the projector using the pinhole model, without distortion

        Args:
            K (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Camera matrices
            pose_camera_in_world (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Extrinsics SE(3) matrix
            h (torch.Tensor, dtype=torch.int64): Image height
            w (torch.Tensor, dtype=torch.int64): Image width
            new_h (int): New height size
            new_w (int): New width size

        Returns:
            None
        """

        # TODO: Add shape checks

        # Get device for later
        device = K.device

        # Initialize pinhole model (no extrinsics)
        E = torch.eye(4).expand(K.shape).to(device)

        # Store original parameters
        self.K = K
        self.height = h
        self.width = w

        new_h = self.height.item() if new_h is None else new_h

        # Compute scale
        sy = new_h / h
        sx = (new_w / w) if (new_w is not None) else sy

        # Compute scaled parameters
        sh = new_h
        sw = new_w if new_w is not None else sh

        # Prepare image cropper
        if new_w is None or new_w == new_h:
            self.image_crop = T.Compose([T.Resize(new_h, T.InterpolationMode.NEAREST), T.CenterCrop(new_h)])
        else:
            self.image_crop = T.Resize([new_h, new_w], T.InterpolationMode.NEAREST)

        # Adjust camera matrix
        # Fill values
        sK = K.clone()
        if new_w is None or new_w == new_h:
            sK[:, 0, 0] = K[:, 1, 1] * sy
            sK[:, 0, 2] = K[:, 1, 2] * sy
            sK[:, 1, 1] = K[:, 1, 1] * sy
            sK[:, 1, 2] = K[:, 1, 2] * sy
        else:
            sK[:, 0, 0] = K[:, 0, 0] * sx
            sK[:, 0, 2] = K[:, 0, 2] * sx
            sK[:, 1, 1] = K[:, 1, 1] * sy
            sK[:, 1, 2] = K[:, 1, 2] * sy

        # Initialize camera with scaled parameters
        sh = torch.IntTensor([sh]).to(device)
        sw = torch.IntTensor([sw]).to(device)
        self.camera = PinholeCamera(sK, E, sh, sw)

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
        self,
        pose_camera_in_world: torch.tensor,
        points: torch.tensor,
        colors: torch.tensor,
        image: torch.tensor = None,
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

    def resize_image(self, image: torch.tensor):
        return self.image_crop(image)


def run_image_projector():
    """Projects 3D points to example images and returns an image with the projection"""

    from wild_visual_navigation.visu import get_img_from_fig
    from wild_visual_navigation.utils import (
        make_polygon_from_points,
    )
    from wild_visual_navigation.utils.testing import load_test_image, make_results_folder
    import matplotlib.pyplot as plt
    import torch
    from kornia.utils import tensor_to_image
    from stego.utils import remove_axes

    # Create test directory
    outpath = make_results_folder("test_image_projector")

    # Define number of cameras (batch)
    B = 10

    # Prepare single pinhole model
    # Camera is created 1.5m backward, and 1m upwards, 0deg towards the origin
    # Intrinsics
    K = torch.FloatTensor([[720, 0, 720, 0], [0, 720, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    K = K.expand(B, 4, 4)

    # Extrisics
    pose_camera_in_world = torch.eye(4).repeat(B, 1, 1)

    for i in range(B):
        rho = torch.FloatTensor([-1.2 - i / 10.0, 0, 1])  # Translation vector (x, y, z)
        phi = torch.FloatTensor([-2 * torch.pi / 4, 0.0, -torch.pi / 2.4])  # roll-pitch-yaw
        R_WC = SO3.from_rpy(phi)  # Rotation matrix from roll-pitch-yaw
        pose_camera_in_world[i] = SE3(R_WC, rho).as_matrix()  # Pose matrix of camera in world frame
    # Image size
    H = torch.tensor(1080)
    W = torch.tensor(1440)

    # Create projector
    im = ImageProjector(K, H, W)

    # Load image
    k_img = load_test_image()
    k_img = k_img.expand(B, 3, H, W)
    k_img = im.resize_image(k_img)

    # Create 3D points around origin
    # X = make_plane(x=0.8, y=0.5, pose=torch.eye(4))
    with Timer("make_plane"):
        # X = make_dense_plane(x=2, y=2, pose=torch.eye(4), grid_size=15)
        points = torch.FloatTensor([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) * 0.5
        X = make_polygon_from_points(points)

    N, D = X.shape
    X = X.expand(B, N, D)
    colors = torch.tensor([0, 1, 0]).expand(B, 3)

    # Project points to image
    with Timer("project_and_render"):
        k_mask, k_img_overlay, k_points, k_valid = im.project_and_render(pose_camera_in_world, X, colors, k_img)

    # Plot points independently
    fig, ax = plt.subplots(B, 4, figsize=(4 * 5, B * 5))

    for i in range(B):
        k_points_overlay = k_img[i].clone()
        for p in k_points[i]:
            idx = torch.round(p).to(torch.int32)
            for y in range(-3, 3, 1):
                for x in range(-3, 3, 1):
                    try:
                        k_points_overlay[:, idx[1].item() + y, idx[0].item() + x] = torch.tensor([0, 255, 0])
                    except Exception:
                        continue

        ax[i, 0].imshow(tensor_to_image(k_img[i]))
        ax[i, 0].set_title("Image")
        ax[i, 1].imshow(tensor_to_image(k_mask[i]))
        ax[i, 1].set_title("Labels")
        ax[i, 2].imshow(tensor_to_image(k_img_overlay[i]))
        ax[i, 2].set_title("Overlay")
        ax[i, 3].imshow(tensor_to_image(k_points_overlay))
        ax[i, 3].set_title("Overlay - dots")
    remove_axes(ax)
    plt.tight_layout()

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(
        join(
            outpath,
            "forest_clean_image_projector.png",
        )
    )


if __name__ == "__main__":
    run_image_projector()
