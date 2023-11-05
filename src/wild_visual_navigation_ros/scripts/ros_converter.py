import cv2
from geometry_msgs.msg import Pose,Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from liegroups.torch import SO3, SE3
import numpy as np
import torch
import torchvision.transforms as transforms
from pytictac import Timer
CV_BRIDGE = CvBridge()
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()
BASE_DIM = 7 + 6  # pose + twist

def ros_tf_to_torch(tf_pose, device="cpu"):
    assert len(tf_pose) == 2
    assert isinstance(tf_pose, tuple)
    if tf_pose[0] is None:
        return False, None
    t = torch.FloatTensor(tf_pose[0])
    q = torch.FloatTensor(tf_pose[1])
    return True, SE3(SO3.from_quaternion(q, ordering="xyzw"), t).as_matrix().to(device)

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion

    # Compute the rotation matrix
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=np.float32)
    return R

def ros_tf_to_numpy(tf_pose):
    assert len(tf_pose) == 2
    assert isinstance(tf_pose, tuple)
    if tf_pose[0] is None:
        return False, None
    t = np.array(tf_pose[0], dtype=np.float32)
    q = np.array(tf_pose[1], dtype=np.float32)

    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    return True, T

def torch_tensor_to_geometry_msgs_PointArray(tensor):
    """
    Converts a torch tensor of shape (n, 3) to a list of ROS geometry_msgs/Point

    Args:
        tensor (torch.Tensor): A torch tensor of shape (n, 3) where each row represents x, y, z coordinates.

    Returns:
        list[geometry_msgs/Point]: A list of geometry_msgs/Point
    """
    # Ensure that the tensor is on the CPU and converted to a numpy array
    points_np = tensor.cpu().detach().numpy()

    # Convert the numpy array to a list of geometry_msgs/Point
    point_list = []
    for point in points_np:
        ros_point = Point(*point)  # Unpack each x, y, z coordinate into the Point constructor
        point_list.append(ros_point)

    return point_list

def np_to_geometry_msgs_PointArray(array):
    """
    Converts a torch tensor of shape (n, 3) to a list of ROS geometry_msgs/Point

    Args:
        tensor (torch.Tensor): A torch tensor of shape (n, 3) where each row represents x, y, z coordinates.

    Returns:
        list[geometry_msgs/Point]: A list of geometry_msgs/Point
    """
    # Ensure that the tensor is on the CPU and converted to a numpy array
    points_np = array

    # Convert the numpy array to a list of geometry_msgs/Point
    point_list = []
    for point in points_np:
        ros_point = Point(*point)  # Unpack each x, y, z coordinate into the Point constructor
        point_list.append(ros_point)

    return point_list