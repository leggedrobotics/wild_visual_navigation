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

    return transform_points(pose, points).squeeze(0)


def make_box(length, width, height, pose=torch.eye(4), grid_size=11):
    r = 0.01
    s = 0.01
    t = 0.01
    return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)


def make_rounded_box(length, width, height, pose=torch.eye(4), grid_size=11):
    r = 0.2
    s = 0.2
    t = 0.2
    return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)


def make_ellipsoid(length, width, height, pose=torch.eye(4), grid_size=11):
    r = 1
    s = 1
    t = 1
    return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)


def make_plane(x=None, y=None, z=None, pose=torch.eye(4), grid_size=10):
    if x is None:
        points = torch.FloatTensor(
            [[0.0, y / 2, z / 2], [0.0, -y / 2, z / 2], [0.0, -y / 2, -z / 2], [0.0, y / 2, -z / 2]]
        )
    elif y is None:
        points = torch.FloatTensor(
            [[x / 2, 0.0, z / 2], [x / 2, 0.0, -z / 2], [-x / 2, 0.0, -z / 2], [-x / 2, 0.0, z / 2]]
        )
    elif z is None:
        points = torch.FloatTensor(
            [[x / 2, y / 2, 0.0], [x / 2, -y / 2, 0.0], [-x / 2, -y / 2, 0.0], [-x / 2, y / 2, 0.0]]
        )
    else:
        raise "make_plane requires just 2 inputs to be set"

    # interpolate according to the gridsize
    finer_points = [points]
    if grid_size > 0:
        w_steps = torch.linspace(0, 1, steps=grid_size)
        for i in range(4):
            for w in w_steps:
                interp = torch.lerp(points[i], points[(i + 1) % 4], w).unsqueeze(0)
                finer_points.append(interp)
    # To torch
    finer_points = torch.cat(finer_points).unsqueeze(0)

    if len(pose.shape) == 2:
        pose = pose.unsqueeze(0)

    return transform_points(pose, finer_points).squeeze(0)


if __name__ == "__main__":
    points = make_plane(x=0.8, y=0.4, grid_size=10)
