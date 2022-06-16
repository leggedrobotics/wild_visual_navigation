from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch
from kornia.geometry.transform import transform
from kornia.geometry.camera.pinhole import PinholeCamera


class ImageProjector:
    r"""
    TODO
    """

    def __init__(self, intrinsics, extrinsics, height, width, fixed_frame, camera_frame):
        """Initializes the projector for B cameras using the pinhole model, without distortion

        Args:
            intrinsics: (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Camera matrices
            extrinsics: (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Extrinsics SE(3) matrices
            height: (torch.Tensor, dtype=torch.int64): Image height
            width: (torch.Tensor, dtype=torch.int64):  Image width
            fixed_frame: (str):  Fixed frame name
            camera_frame: (str): Camera frame name

        Returns:
            None
        """

        # Get size of the batch
        N = intrinsics.shape[0]
        M = extrinsics.shape[0]

        # Initialize pinhole model
        self.camera = PinholeCamera(intrinsics, extrinsics, height.expand(N), width.expand(N))
        self.fixed_frame = fixed_frame
        self.camera_frame = camera_frame

    def project(self, points, points_frame):
        """Applies the pinhole projection model to a batch of points

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, 1, 3)): Batch of input points on 3D space
            points_frame: (str): Frame used to express the input points

        Returns:
            projected_points: (torch.Tensor, dtype=torch.float32, shape=(B, 1, 2)): Batch of input points on image space
        """

        if self.fixed_frame != points_frame:
            print(f"Input points frame [{points_frame}] doesn't match the frame of the camera [{self.fixed_frame}]")
            raise

        # Apply projection
        projected_points = self.camera.project(points)

        # projected frame
        projected_frame = self.camera_frame

        return projected_points, projected_frame

    def project_and_render(self, points, points_frame):
        """Projects the points and returns an image with the projection

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, 1, 3)): Batch of input points on 3D space
            points_frame: (str): Frame used to express the input points

        Returns:
            out_img (torch.tensor, dtype=torch.int64): Image with projected points
        """
        # This should execute the function above to generate 2d points on the image plane
        # then it should find a convex hull
        # finally should inpaint the convex hull and generate a (B, 1, H, W) mask (1 channel)
        pass


def run_image_projector():
    """Projects 3D points to example images and returns an image with the projection"""

    # from PIL import Image
    # from wild_visual_navigation.utils import get_img_from_fig
    # import matplotlib.pyplot as plt
    # from stego.src import unnorm, remove_axes
    # import numpy as np

    # # Create test directory
    # os.makedirs(join(WVN_ROOT_DIR, "results", "test_image_projector"), exist_ok=True)

    # # Inference model
    # si = StegoInterface(device="cuda")
    # img = Image.open(join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    # linear_pred, cluster_pred = si.inference(img)

    # # Plot result as in colab
    # fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    # ax[0].imshow(unnorm(si.transform(img)).permute(1, 2, 0))
    # ax[0].set_title("Image")
    # ax[1].imshow(si.model.label_cmap[cluster_pred])
    # ax[1].set_title("Cluster Predictions")
    # ax[2].imshow(si.model.label_cmap[linear_pred])
    # ax[2].set_title("Linear Probe Predictions")
    # remove_axes(ax)

    # # Store results to test directory
    # img = get_img_from_fig(fig)
    # img.save(join(WVN_ROOT_DIR, "results", "test_stego_interfacer", "forest_clean_stego.png"))


if __name__ == "__main__":
    run_image_projector()
