#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
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
    eta_s = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=grid_size).to(pose.device)
    w_s = torch.linspace(-torch.pi, torch.pi, steps=grid_size).to(pose.device)
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

    x = x.ravel()[:, None]
    y = y.ravel()[:, None]
    z = z.ravel()[:, None]
    points = torch.cat((x, y, z), dim=1)[None]

    # Apply pose transformation
    if len(pose.shape) == 2:
        pose = pose[None]

    return transform_points(pose, points)


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
            [
                [0.0, y / 2, z / 2],
                [0.0, -y / 2, z / 2],
                [0.0, -y / 2, -z / 2],
                [0.0, y / 2, -z / 2],
            ]
        ).to(pose.device)
    elif y is None:
        points = torch.FloatTensor(
            [
                [x / 2, 0.0, z / 2],
                [x / 2, 0.0, -z / 2],
                [-x / 2, 0.0, -z / 2],
                [-x / 2, 0.0, z / 2],
            ]
        ).to(pose.device)
    elif z is None:
        points = torch.FloatTensor(
            [
                [x / 2, y / 2, 0.0],
                [x / 2, -y / 2, 0.0],
                [-x / 2, -y / 2, 0.0],
                [-x / 2, y / 2, 0.0],
            ]
        ).to(pose.device)
    else:
        raise "make_plane requires just 2 inputs to be set"

    # Interpolate according to the gridsize
    finer_points = [points]
    if grid_size > 0:
        w_steps = torch.linspace(0, 1, steps=grid_size).to(pose.device)
        for i in range(4):
            for w in w_steps:
                interp = torch.lerp(points[i], points[(i + 1) % 4], w)[None]
                finer_points.append(interp)

    # To torch
    finer_points = torch.cat(finer_points)
    finer_points = torch.unique(finer_points, dim=0)

    if len(pose.shape) == 2:
        pose = pose[None]

    return transform_points(pose, finer_points[None])[0]


def make_dense_plane(x=None, y=None, z=None, pose=torch.eye(4), grid_size=5):
    if x is None:
        x_s = torch.linspace(0.0, 0.0, steps=grid_size).to(pose.device)
        y_s = torch.linspace(-y / 2, y / 2, steps=grid_size).to(pose.device)
        z_s = torch.linspace(-z / 2, z / 2, steps=grid_size).to(pose.device)
    elif y is None:
        x_s = torch.linspace(-x / 2, x / 2, steps=grid_size).to(pose.device)
        y_s = torch.linspace(0.0, 0.0, steps=grid_size).to(pose.device)
        z_s = torch.linspace(-z / 2, z / 2, steps=grid_size).to(pose.device)
    elif z is None:
        x_s = torch.linspace(-x / 2, x / 2, steps=grid_size).to(pose.device)
        y_s = torch.linspace(-y / 2, y / 2, steps=grid_size).to(pose.device)
        z_s = torch.linspace(0.0, 0.0, steps=grid_size).to(pose.device)
    else:
        raise "make_plane requires just 2 inputs to be set"

    x, y, z = torch.meshgrid(x_s, y_s, z_s, indexing="xy")

    x = x.ravel()[:, None]
    y = y.ravel()[:, None]
    z = z.ravel()[:, None]
    points = torch.cat((x, y, z), dim=1)

    if len(pose.shape) == 2:
        pose = pose[None]

    return transform_points(pose, points[None])[0]


def make_polygon_from_points(points: torch.tensor, grid_size=10):
    B, D = points.shape
    finer_points = []
    w_steps = torch.linspace(0, 1, steps=grid_size).to(points.device)
    # assume the points are sorted
    for i in range(B):
        for w in w_steps:
            finer_points.append(torch.lerp(points[i], points[(i + 1) % B], w)[None])
    finer_points = torch.cat(finer_points, dim=0)
    return finer_points


if __name__ == "__main__":
    xy_plane = make_dense_plane(x=0.8, y=0.4, grid_size=3)
    y_points = make_plane(x=0.0, y=0.4, grid_size=2)

    points = torch.FloatTensor([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    polygon = make_polygon_from_points(points)
