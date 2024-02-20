from wild_visual_navigation.visu import get_img_from_fig
from wild_visual_navigation.utils.testing import make_results_folder
from os.path import join
import matplotlib.pyplot as plt
import torch
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.utils import draw_convex_polygon, tensor_to_image
from liegroups.torch import SE3


def test_pinhole_camera():
    # Prepare single pinhole model
    # Intrinsics
    K = torch.FloatTensor([[720, 0, 720, 0], [0, 720, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Extrinsics
    xi = torch.FloatTensor([0, 0, 0, 0, 0, 0])  # (x, y, z, rx, ry, rz)
    pose_camera_in_world = SE3.exp(xi).as_matrix()
    T_CW = pose_camera_in_world.inverse()
    T_CW = T_CW.unsqueeze(0)
    # Image size
    H = torch.IntTensor([1080])
    W = torch.IntTensor([1440])

    # Create camera
    camera1 = PinholeCamera(K, T_CW, H, W)

    # Test projection
    # Point in front of camera
    X1 = torch.FloatTensor([0, 0, 1]).unsqueeze(0)
    x1 = camera1.project(X1)
    print(X1, x1)

    # Point behind the camera
    X2 = torch.FloatTensor([0, 0, -1]).unsqueeze(0)
    x2 = camera1.project(X2)
    print(X2, x2)


def test_draw_polygon():
    # Create blank RGB image
    img = torch.zeros((3, 480, 640), dtype=torch.float32).unsqueeze(0)
    # Create list of 2d points to make a polygon
    poly = torch.tensor([[[50, 100], [100, 130], [130, 70], [70, 50]]])
    # Color
    color = torch.tensor([[0.5, 1.0, 0.5]])
    # Draw
    k_out = draw_convex_polygon(img, poly, color)
    # Show
    outpath = make_results_folder("test_kornia")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(tensor_to_image(k_out))
    out_img = get_img_from_fig(fig)
    out_img.save(join(outpath, "polygon_test.png"))
