from wild_visual_navigation.image_projector import run_image_projector
from wild_visual_navigation.traversability_estimator import SupervisionNode


def test_image_projector():
    run_image_projector()


def test_supervision_projection():
    from wild_visual_navigation.image_projector import ImageProjector
    from wild_visual_navigation.visu import get_img_from_fig
    from wild_visual_navigation.utils.testing import load_test_image, make_results_folder
    from liegroups.torch import SE3, SO3
    import matplotlib.pyplot as plt
    import torch
    from os.path import join
    from kornia.utils import tensor_to_image
    from stego.utils import remove_axes
    import random

    # Create test directory
    outpath = make_results_folder("test_image_projector")

    # Define number of cameras (batch)
    B = 100

    # Prepare single pinhole model
    # Camera is created 1.5m backward, and 1m upwards, 0deg towards the origin
    # Intrinsics
    K = torch.FloatTensor([[720, 0, 720, 0], [0, 720, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])[None]

    # Extrisics
    pose_camera_in_world = torch.eye(4)[None]
    # Image size
    H = torch.tensor(1080)
    W = torch.tensor(1440)

    # Create projector
    im = ImageProjector(K, H, W)

    # Load image
    torch_image = load_test_image()

    # Resize
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
        # print(delta, pose_base_in_world)

        twist = torch.rand((3,))
        supervision = torch.rand((10,))
        traversability = torch.rand(1)
        traversability_var = torch.tensor([0.2])
        color = torch.rand((3,))[None]

        supervision_node = SupervisionNode(
            timestamp=i,
            pose_base_in_world=pose_base_in_world,
            pose_footprint_in_base=pose_footprint_in_base,
            twist_in_base=twist,
            width=0.5,
            length=0.8,
            height=0.1,
            supervision=supervision,
            traversability=traversability,
            traversability_var=traversability_var,
            is_untraversable=torch.BoolTensor([False]),
        )
        nodes.append(supervision_node)

        if i > 0:
            footprint = supervision_node.make_footprint_with_node(nodes[i - 1])[None]

            # project footprint
            k_mask, torch_image_overlay, k_points, k_valid = im.project_and_render(
                pose_camera_in_world, footprint, color
            )

            mask = torch.fmax(mask, k_mask)

    # Plot points independently
    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 5))
    ax[0].imshow(tensor_to_image(torch_image))
    ax[0].set_title("Image")
    ax[1].imshow(tensor_to_image(mask[0]))
    ax[1].set_title("Labels")

    remove_axes(ax)
    plt.tight_layout()

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(join(outpath, "forest_clean_supervision_projection.png"))


if __name__ == "__main__":
    test_supervision_projection()
