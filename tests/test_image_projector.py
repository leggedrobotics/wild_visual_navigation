from wild_visual_navigation.image_projector import run_image_projector
from wild_visual_navigation.traversability_estimator import ProprioceptionNode


def test_image_projector():
    run_image_projector()


def test_proprioceptive_projection():
    from wild_visual_navigation import WVN_ROOT_DIR
    from wild_visual_navigation.image_projector import ImageProjector
    from wild_visual_navigation.visu import get_img_from_fig
    from wild_visual_navigation.utils import Timer
    from wild_visual_navigation.utils import make_plane, make_box, make_dense_plane, make_polygon_from_points
    from PIL import Image
    from liegroups.torch import SE3, SO3
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms as transforms
    import os
    from os.path import join
    from kornia.utils import tensor_to_image
    from stego.src import remove_axes
    import random

    to_tensor = transforms.ToTensor()

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_image_projector"), exist_ok=True)

    # Define number of cameras (batch)
    B = 100

    # Prepare single pinhole model
    # Camera is created 1.5m backward, and 1m upwards, 0deg towards the origin
    # Intrinsics
    K = torch.FloatTensor([[720, 0, 720, 0], [0, 720, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])[None]

    # Extrisics
    pose_camera_in_world = torch.eye(4)[None]

    # Image size
    H = 1080
    W = 1440

    # Create projector
    im = ImageProjector(K, H, W)

    # Load image
    pil_image = Image.open(join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))

    # Convert to torch
    torch_image = to_tensor(pil_image)
    torch_image = im.resize_image(torch_image)
    mask = (torch_image * 0.0)[None]

    rho = torch.FloatTensor([0, 0, 2])  # Translation vector (x, y, z)
    # phi = torch.FloatTensor([-2 * torch.pi / 4, 0.0, -torch.pi / 2])  # roll-pitch-yaw
    phi = torch.FloatTensor([-3 * torch.pi / 4, 0.0, -torch.pi / 2])  # roll-pitch-yaw
    R_WC = SO3.from_rpy(phi)  # Rotation matrix from roll-pitch-yaw
    pose_camera_in_world = SE3(R_WC, rho).as_matrix()[None]  # Pose matrix of camera in world frame

    # Fill data
    pose_base_in_world = torch.eye(4)[None]
    nodes = []
    for i in range(B):
        rho = torch.FloatTensor([1 / 10.0 + random.random() / 10.0, 0, 0])  # Translation vector (x, y, z)
        phi = torch.FloatTensor([0.0, 0.0, (random.random() - 0.5) / 2])  # roll-pitch-yaw
        R_WC = SO3.from_rpy(phi)  # Rotation matrix from roll-pitch-yaw
        delta = SE3(R_WC, rho).as_matrix()[None]  # Pose matrix of camera in world frame
        pose_base_in_world = pose_base_in_world @ delta
        pose_footprint_in_base = torch.eye(4)[None]
        print(delta, pose_base_in_world)

        twist = torch.rand((3,))
        proprioception = torch.rand((10,))
        traversability = torch.rand(1)
        traversability_var = torch.tensor([0.2])
        color = torch.rand((3,))[None]

        proprio_node = ProprioceptionNode(
            timestamp=i,
            pose_base_in_world=pose_base_in_world,
            pose_footprint_in_base=pose_footprint_in_base,
            twist_in_base=twist,
            width=0.5,
            length=0.8,
            height=0.1,
            proprioception=proprioception,
            traversability=traversability,
            traversability_var=traversability_var,
            is_untraversable=torch.BoolTensor([False]),
        )
        nodes.append(proprio_node)

        if i > 0:
            footprint = proprio_node.make_footprint_with_node(nodes[i - 1])[None]

            # project footprint
            k_mask, torch_image_overlay, k_points, k_valid = im.project_and_render(
                pose_camera_in_world, footprint, color
            )

            mask = torch.fmax(mask, k_mask)

    # Plot points independently
    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 5))
    ax[0].imshow(tensor_to_image(torch_image))
    ax[0].set_title("Image")
    ax[1].imshow(tensor_to_image(mask))
    ax[1].set_title("Labels")

    remove_axes(ax)
    plt.tight_layout()

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(join(WVN_ROOT_DIR, "results", "test_image_projector", "forest_clean_proprioceptive_projection.png"))


if __name__ == "__main__":
    test_proprioceptive_projection()
