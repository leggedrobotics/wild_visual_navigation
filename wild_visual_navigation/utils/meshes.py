import torch
from kornia.geometry.linalg import transform_points


def make_superquadric(A, B, C, r, s, t, pose=torch.eye(4), grid_size=10):
    """Returns a set of 3D points representing a superquadric given by the
    shape parameters at the specified pose

    Args:
        A: (float): Size along x
        B: (float): Size along y
        C: (float): Size along z
        r: (float): Exponent for x
        s: (float): Exponent for y
        t: (float): Exponent for z
        pose: (torch.Tensor, dtype=torch.float32, shape=(4, 4)): SE(3) pose
        grid_size: (int): discretization of the surface

    Returns:
        out_img (torch.tensor, dtype=torch.int64): Image with projected points
    """

    # Prepare meshgrid
    eta_s = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=grid_size)
    w_s = torch.linspace(-torch.pi, torch.pi, steps=grid_size)
    eta, w = torch.meshgrid(eta_s, w_s, indexing="xy")

    # Compute coordinates
    cos_eta = torch.cos(eta)
    sin_eta = torch.sin(eta)
    cos_w = torch.cos(w)
    sin_w = torch.sin(w)

    # Compute superquadric
    x = A * cos_eta.sign() * cos_eta.abs().pow(r) * cos_w.sign() * cos_w.abs().pow(r)
    y = B * cos_eta.sign() * cos_eta.abs().pow(s) * sin_w.sign() * sin_w.abs().pow(s)
    z = C * sin_eta.sign() * sin_eta.abs().pow(s)

    x = x.ravel().unsqueeze(1)
    y = y.ravel().unsqueeze(1)
    z = z.ravel().unsqueeze(1)
    points = torch.cat((x, y, z), dim=1).unsqueeze(0)

    # Apply pose transformation
    if len(pose.shape) == 2:
        pose = pose.unsqueeze(0)

    return transform_points(pose, points)


def make_box(length, width, height, pose=torch.eye(4), grid_size=11):
    r = 0.01
    s = 0.01
    t = 0.01
    return make_superquadric(length, width, height, r, s, t, pose=pose, grid_size=grid_size)


def make_rounded_box(length, width, height, pose=torch.eye(4), grid_size=11):
    r = 0.2
    s = 0.2
    t = 0.2
    return make_superquadric(length, width, height, r, s, t, pose=pose, grid_size=grid_size)


def make_ellipsoid(length, width, height, pose=torch.eye(4), grid_size=11):
    r = 1
    s = 1
    t = 1
    return make_superquadric(length, width, height, r, s, t, pose=pose, grid_size=grid_size)
