import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, Image
from grid_map_msgs.msg import GridMap, GridMapInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import TransformStamped
import numpy as np
import ros_numpy


class SimpleNumpyToRviz:
    def __init__(self, init_node=True, postfix=""):
        rospy.init_node("numpy_to_rviz", anonymous=False)
        self.pub_pointcloud = rospy.Publisher(f"~pointcloud{postfix}", PointCloud2, queue_size=1)
        self.pub_gridmap = rospy.Publisher(f"~gridmap{postfix}", GridMap, queue_size=1)
        self.pub_image = rospy.Publisher(f"~image{postfix}", Image, queue_size=1)
        self.pub_camera_info = rospy.Publisher(f"~camera_info{postfix}", CameraInfo, queue_size=1)
        self.br = tf2_ros.TransformBroadcaster()

    def camera_info(self, camera_info):
        pass

    def tf(self, msg, reference_frame="crl_rzr/map"):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = reference_frame
        t.child_frame_id = msg[3]
        t.transform.translation.x = msg[4][0]
        t.transform.translation.y = msg[4][1]
        t.transform.translation.z = msg[4][2]
        t.transform.rotation.x = msg[5][0]
        t.transform.rotation.y = msg[5][1]
        t.transform.rotation.z = msg[5][2]
        t.transform.rotation.w = msg[5][3]
        self.br.sendTransform(t)

    def image(self, img):
        pass

    def pointcloud(self, points, reference_frame="sensor_gravity"):
        data = np.zeros(points.shape[0], dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]
        msg = ros_numpy.msgify(PointCloud2, data)

        msg.header.frame_id = reference_frame

        self.pub_pointcloud.publish(msg)

    def gridmap(self, msg, publish=True):
        data_in = msg[0]

        size_x = data_in["data"].shape[1]
        size_y = data_in["data"].shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        layers = []
        data = []

        for i in range(data_in["data"].shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = data_in["data"][i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.position.x = data_in["position"][0]
        info.pose.position.y = data_in["position"][1]
        info.pose.position.z = data_in["position"][2]
        info.pose.orientation.x = data_in["orientation_xyzw"][0]
        info.pose.orientation.y = data_in["orientation_xyzw"][1]
        info.pose.orientation.z = data_in["orientation_xyzw"][2]
        info.pose.orientation.z = data_in["orientation_xyzw"][3]
        info.header.seq = msg[1]
        # info.header.stamp.secs = msg[2]vis
        info.header.stamp = rospy.Time.now()
        info.resolution = data_in["resolution"]
        info.length_x = size_x * data_in["resolution"]
        info.length_y = size_y * data_in["resolution"]

        gm_msg = GridMap(info=info, layers=data_in["layers"], basic_layers=data_in["basic_layers"], data=data)

        if publish:
            self.pub_gridmap.publish(gm_msg)

        return gm_msg

    def gridmap_arr(self, arr, res, layers, reference_frame="sensor_gravity", publish=True, x=0, y=0):
        size_x = arr.shape[1]
        size_y = arr.shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data = []

        for i in range(arr.shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = arr[i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.orientation.w = 1
        info.header.seq = 0
        info.header.stamp = rospy.Time.now()
        info.resolution = res
        info.length_x = size_x * res
        info.length_y = size_y * res
        info.header.frame_id = reference_frame
        info.pose.position.x = x
        info.pose.position.y = y
        gm_msg = GridMap(info=info, layers=layers, basic_layers=[], data=data)
        if publish:
            self.pub_gridmap.publish(gm_msg)
        return gm_msg


if __name__ == "__main__":
    import time
    import pickle as pkl
    from os.path import join
    import torch

    vis = SimpleNumpyToRviz()
