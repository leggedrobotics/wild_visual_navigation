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