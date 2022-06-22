import torch
from kornia.geometry.linalg import transform_points


def box_constraint(tensor, h_length, h_width, h_height, eps=0.1):
    l_pos = (tensor - h_length).abs() < eps
    l_neg = (tensor + h_length).abs() < eps
    w_pos = (tensor - h_width).abs() < eps
    w_neg = (tensor + h_width).abs() < eps
    h_pos = (tensor - h_height).abs() < eps
    h_neg = (tensor + h_height).abs() < eps
    return (l_pos | l_neg | w_pos | w_neg | h_pos | h_neg).any(1)


def ellipsoid_constraint(tensor, h_length, h_width, h_height, eps=0.1):
    return (
        torch.pow(tensor[:, 0] / (h_length + 1e-6), 2)
        + torch.pow(tensor[:, 1] / (h_width + 1e-6), 2)
        + torch.pow(tensor[:, 2] / (h_height + 1e-6), 2)
        - 1
    ).abs() < eps


def make_polytope(constraint, pose, x_size, y_size, z_size, grid_size=10, eps=1e-6):
    half_x_size = x_size / 2
    half_y_size = y_size / 2
    half_z_size = z_size / 2

    xs = torch.linspace(-half_x_size, half_x_size, steps=grid_size)
    ys = torch.linspace(-half_y_size, half_y_size, steps=grid_size)
    zs = torch.linspace(-half_z_size, half_z_size, steps=grid_size)

    # Points
    xyz = torch.cartesian_prod(xs, ys, zs)

    # Get mask of all the points that satisfy the constraint
    mask = constraint(xyz, x_size, y_size, z_size, eps)

    # Mask points
    points = xyz[mask].unsqueeze(0)

    # Apply pose transformation
    pose = pose.unsqueeze(0)
    return transform_points(pose, points)


def make_box(pose, length, width, height, grid_size=10, eps=0.1):
    return make_polytope(box_constraint, pose, length, width, height, grid_size, eps)


def make_ellipsoid(pose, length, width, height, grid_size=10, eps=0.1):
    return make_polytope(ellipsoid_constraint, pose, length, width, height, grid_size, eps)


# def make_box(pose, length, width, height, grid_size=10):
#     """Projects the points and returns an image with the projection

#         Args:
#             pose: (torch.Tensor, dtype=torch.float32, shape=(4, 4)): Input SE(3) matrix
#             length: (float): Size along x
#             width:  (float): Size along y
#             height: (float): Size along z
#             grid_size: (int): Resolution of the polytope

#         Returns:
#             out_img (torch.tensor, dtype=torch.int64): Image with projected points
#     """

#     xs = torch.linspace(-length/2, length/2, steps=grid_size)
#     ys = torch.linspace(-width/2, width/2, steps=grid_size)
#     zs = torch.linspace(-height/2, height/2, steps=grid_size)

#     # Points
#     xyz = torch.cartesian_prod(xs, ys, zs)

#     # Get mask of all the points that satisfy the constraint
#     mask = box_constraint(xyz)

#     # Return only values that
#     return transform_points(pose, xyz[mask])


# def make_ellipsoid(pose, length, width, height, grid_size=10):
#     """Projects the points and returns an image with the projection

#         Args:
#             pose: (torch.Tensor, dtype=torch.float32, shape=(4, 4)): Input SE(3) matrix
#             length: (float): Size along x
#             width:  (float): Size along y
#             height: (float): Size along z
#             grid_size: (int): Resolution of the polytope

#         Returns:
#             out_img (torch.tensor, dtype=torch.int64): Image with projected points
#     """

#     xs = torch.linspace(-length/2, length/2, steps=grid_size)
#     ys = torch.linspace(-width/2, width/2, steps=grid_size)
#     zs = torch.linspace(-height/2, height/2, steps=grid_size)

#     # Points
#     xyz = torch.cartesian_prod(xs, ys, zs)

#     def ellipsoid_constraint(tensor):
#         eps = 1e-6
#         l_pos = (xyz -  length/2).abs() < eps
#         l_neg = (xyz - -length/2).abs() < eps
#         w_pos = (xyz -  length/2).abs() < eps
#         w_neg = (xyz - -length/2).abs() < eps
#         h_pos = (xyz -  length/2).abs() < eps
#         h_neg = (xyz - -length/2).abs() < eps
#         return (l_pos | l_neg | w_pos | w_neg | h_pos | h_neg).any(1)

#     # Get mask of all the points that satisfy the constraints
#     mask = ellipse_constraint(xyz)

#     # Return only values that
#     return transform_points(pose, xyz[mask])
