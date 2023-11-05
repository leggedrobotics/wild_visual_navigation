import torch
from kornia.geometry.linalg import transform_points
from numba import jit
import numpy as np

# def make_superquadric(A, B, C, r, s, t, pose=torch.eye(4), grid_size=10):
#     """Returns a set of 3D points representing a superquadric given by the
#     shape parameters at the specified pose

#     Args:
#         A: (float): Size along x
#         B: (float): Size along y
#         C: (float): Size along z
#         r: (float): Exponent for x
#         s: (float): Exponent for y
#         t: (float): Exponent for z
#         pose: (torch.Tensor, dtype=torch.float32, shape=(4, 4)): SE(3) pose
#         grid_size: (int): discretization of the surface

#     Returns:
#         out_img (torch.tensor, dtype=torch.int64): Image with projected points
#     """
#     if C == 0:
#         # Generating a 2D ellipse on the x-y plane
#         w_s = torch.linspace(-torch.pi, torch.pi, steps=grid_size).to(pose.device)
#         cos_w = torch.cos(w_s)
#         sin_w = torch.sin(w_s)

#         x = A * cos_w.sign() * cos_w.abs().pow(r)
#         y = B * sin_w.sign() * sin_w.abs().pow(s)
#         z = torch.zeros_like(x)

#         points = torch.stack((x, y, z), dim=1)[None]
#     else:
#         # Prepare meshgrid
#         eta_s = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=grid_size).to(pose.device)
#         w_s = torch.linspace(-torch.pi, torch.pi, steps=grid_size).to(pose.device)
#         eta, w = torch.meshgrid(eta_s, w_s, indexing="xy")

#         # Compute coordinates
#         cos_eta = torch.cos(eta)
#         sin_eta = torch.sin(eta)
#         cos_w = torch.cos(w)
#         sin_w = torch.sin(w)

#         # Compute superquadric
#         x = A * cos_eta.sign() * cos_eta.abs().pow(r) * cos_w.sign() * cos_w.abs().pow(r)
#         y = B * cos_eta.sign() * cos_eta.abs().pow(s) * sin_w.sign() * sin_w.abs().pow(s)
#         z = C * sin_eta.sign() * sin_eta.abs().pow(t)

#         x = x.ravel()[:, None]
#         y = y.ravel()[:, None]
#         z = z.ravel()[:, None]
#         points = torch.cat((x, y, z), dim=1)[None]

#     # Apply pose transformation
#     if len(pose.shape) == 2:
#         pose = pose[None]

#     return transform_points(pose, points)[0]


# def make_box(length, width, height, pose=torch.eye(4), grid_size=11):
#     r = 0.01
#     s = 0.01
#     t = 0.01
#     return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)


# def make_rounded_box(length, width, height, pose=torch.eye(4), grid_size=11):
#     r = 0.2
#     s = 0.2
#     t = 0.2
#     return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)


# def make_ellipsoid(length, width, height, pose=torch.eye(4), grid_size=11):
#     r = 1
#     s = 1
#     t = 1
#     return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)
@jit(nopython=True)
def make_superquadric(A, B, C, r, s, t, pose, grid_size=10):
    if pose is None:
        pose = np.eye(4, dtype=np.float32)
    if C == 0:
        # Generating a 2D ellipse on the x-y plane
        w_s = np.linspace(-np.pi, np.pi, grid_size)
        cos_w = np.cos(w_s)
        sin_w = np.sin(w_s)

        x = A * np.sign(cos_w) * np.abs(cos_w) ** r
        y = B * np.sign(sin_w) * np.abs(sin_w) ** s
        z = np.zeros_like(x)

        points = np.stack((x, y, z), axis=1)
    else:
        # Prepare meshgrid
        eta_s = np.linspace(-np.pi / 2, np.pi / 2, grid_size)
        w_s = np.linspace(-np.pi, np.pi, grid_size)
        # eta, w = np.meshgrid(eta_s, w_s, indexing="xy")
        eta, w = custom_meshgrid(eta_s, w_s)
        # Compute coordinates
        cos_eta = np.cos(eta)
        sin_eta = np.sin(eta)
        cos_w = np.cos(w)
        sin_w = np.sin(w)

        # Compute superquadric
        x = A * np.sign(cos_eta) * np.abs(cos_eta) ** r * np.sign(cos_w) * np.abs(cos_w) ** r
        y = B * np.sign(cos_eta) * np.abs(cos_eta) ** s * np.sign(sin_w) * np.abs(sin_w) ** s
        z = C * np.sign(sin_eta) * np.abs(sin_eta) ** t

        x = x.ravel()[:, None]
        y = y.ravel()[:, None]
        z = z.ravel()[:, None]
        points = np.concatenate((x, y, z), axis=1)
        
    # Apply pose transformation
    # Assuming 'transform_points' is a function that takes numpy arrays as well
    return np_transform_points(pose, points)
@jit(nopython=True)
def custom_meshgrid(x, y):
    m, n = len(x), len(y)
    xx = np.empty((m, n), dtype=x.dtype)
    yy = np.empty((m, n), dtype=y.dtype)

    for i in range(m):
        for j in range(n):
            xx[i, j] = x[i]
            yy[i, j] = y[j]
    
    return xx, yy
def make_box(length, width, height, pose, grid_size=11):
    r = 0.01
    s = 0.01
    t = 0.01
    return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)

def make_rounded_box(length, width, height, pose, grid_size=11):
    r = 0.2
    s = 0.2
    t = 0.2
    return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)

def make_ellipsoid(length, width, height, pose, grid_size=11):
    r = 1
    s = 1
    t = 1
    return make_superquadric(length / 2, width / 2, height / 2, r, s, t, pose=pose, grid_size=grid_size)


def make_plane(x=None, y=None, z=None, pose=np.eye(4), grid_size=10):
    simple = True
    if x is None:
        points = np.array(
            [[0.0, y / 2, z / 2], [0.0, -y / 2, z / 2], [0.0, -y / 2, -z / 2], [0.0, y / 2, -z / 2]]
        )
    elif y is None:
        points = np.array(
            [[x / 2, 0.0, z / 2], [x / 2, 0.0, -z / 2], [-x / 2, 0.0, -z / 2], [-x / 2, 0.0, z / 2]]
        )
    elif z is None:
        if x != 0:
            points = np.array(
                [[x / 2, y / 2, 0.0], [x / 2, -y / 2, 0.0], [-x / 2, -y / 2, 0.0], [-x / 2, y / 2, 0.0]]
            )
        else:
            points = np.array(
                [[x / 2, y / 2, 0.0], [x / 2, -y / 2, 0.0]]
            )
    else:
        raise ValueError("make_plane requires just 2 inputs to be set")

    if not simple:
        # Interpolate according to the grid size
        finer_points = [points]
        if grid_size > 0:
            w_steps = np.linspace(0, 1, num=grid_size)
            for i in range(4):
                for w in w_steps:
                    interp = np.expand_dims(np.interp(w, [0, 1], [points[i], points[(i + 1) % 4]]), axis=0)
                    finer_points.append(interp)

        # To numpy
        finer_points = np.concatenate(finer_points)
        finer_points = np.unique(finer_points, axis=0)
    else:
        finer_points = points

    return np_transform_points(pose, finer_points)


# def np_transform_points(pose, points):
#     # Assuming the transform_points function is for applying a transformation matrix to a set of points
#     # Here's a simple version that applies a transformation matrix to each point
#     transformed_points = []
#     for point in points:
#         # Homogeneous coordinates
#         homogeneous_point = np.append(point, 1)
#         transformed_point = pose @ homogeneous_point
#         # transformed_point=np.matmul(pose,homogeneous_point)
#         # Discard the homogeneous coordinate before appending
#         transformed_points.append(transformed_point[:-1])
#     return np.array(transformed_points)
@jit(nopython=True)
def np_transform_points(pose, points):
    # Ensure the data types are consistent
    pose = np.asarray(pose, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)

    # Preallocate the array with the appropriate size
    transformed_points = np.empty((points.shape[0], pose.shape[1] - 1), dtype=np.float32)

    for i in range(points.shape[0]):
        homogeneous_point = np.ones(pose.shape[0], dtype=np.float32)
        homogeneous_point[:-1] = points[i]

        # Perform matrix multiplication
        transformed_point = np.dot(pose, homogeneous_point)

        # Assign the result to the preallocated array
        transformed_points[i, :] = transformed_point[:-1]

    return transformed_points

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
