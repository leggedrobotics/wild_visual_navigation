import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import math
import time

import rospy
from grid_map_msgs.msg import GridMap, GridMapInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from dynamic_reconfigure.server import Server
import tf2_ros


class SmartCarrotNode:
    def __init__(self):
        # ROS topics
        self.gridmap_sub_topic = rospy.get_param("~gridmap_sub_topic")
        self.debug_pub_topic = rospy.get_param("~debug_pub_topic")
        self.goal_pub_topic = rospy.get_param("~goal_pub_topic")
        self.map_frame = rospy.get_param("~map_frame")
        self.base_frame = rospy.get_param("~base_frame")
        # Operation Mode
        self.debug = rospy.get_param("~debug")

        # Parameters
        self.distance_force_factor = rospy.get_param("~distance_force_factor")
        self.center_force_factor = rospy.get_param("~center_force_factor")

        # TODO this could be implemented
        self.filter_chain_funcs = []
        if self.distance_force_factor > 0:
            self.filter_chain_funcs.append(self.apply_distance_force)
            self.distance_force = None

        if self.center_force_factor > 0:
            self.filter_chain_funcs.append(self.apply_center_force)

            def distance_to_line(array, x_cor, y_cor, yaw, start_x, start_y):
                return np.abs(np.cos(yaw) * (x_cor - start_x) - np.sin(yaw) * (y_cor - start_y))

            self.vdistance_to_line = np.vectorize(distance_to_line)

        # Initialize ROS publishers
        self.pub = rospy.Publisher(f"~{self.debug_pub_topic}", GridMap, queue_size=5)
        self.pub_goal = rospy.Publisher(self.goal_pub_topic, PoseWithCovarianceStamped, queue_size=5)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.sub = rospy.Subscriber(self.gridmap_sub_topic, GridMap, self.callback, queue_size=5)

    def apply_distance_force(self, yaw, sdf):
        if self.distance_force is None:
            # Create the distance force
            self.distance_force = np.zeros((sdf.shape[0], sdf.shape[1]))
            for x in range(sdf.shape[0]):
                for y in range(sdf.shape[1]):
                    self.distance_force[x, y] = math.sqrt(
                        (x - int(sdf.shape[0] / 2)) ** 2 + (y - int(sdf.shape[1] / 2)) ** 2
                    )
            self.distance_force /= self.distance_force.max()
            self.distance_force *= self.distance_force_factor
        return sdf + self.distance_force

    def apply_center_force(self, yaw, sdf):
        xv, yv = np.meshgrid(np.arange(0, sdf.shape[0]), np.arange(0, sdf.shape[1]))
        center_force = self.vdistance_to_line(sdf, xv, yv, yaw, int(sdf.shape[0] / 2), int(sdf.shape[1] / 2))
        return sdf - center_force * self.center_force_factor

    def get_pattern_mask(self, H, W, yaw):
        # Defines a pattern based on the yaw of the robot where we search for a minimum within the SDF
        binary_mask = np.zeros((H, W), dtype=np.uint8)
        distance = 30
        center_x = int(H / 2 + math.sin(yaw) * distance)
        center_y = int(W / 2 + math.cos(yaw) * distance)
        binary_mask = cv2.circle(binary_mask, (center_x, center_y), 0, 255, 30)
        distance = 55
        center_x = int(H / 2 + math.sin(yaw) * distance)
        center_y = int(W / 2 + math.cos(yaw) * distance)
        binary_mask = cv2.circle(binary_mask, (center_x, center_y), 0, 255, 40)
        distance = 90
        center_x = int(H / 2 + math.sin(yaw) * distance)
        center_y = int(W / 2 + math.cos(yaw) * distance)
        binary_mask = cv2.circle(binary_mask, (center_x, center_y), 0, 255, 50)

        return binary_mask == 0

    def get_elevation_mask(self, elevation_layer):
        invalid_elevation = np.isnan(elevation_layer)
        # Increase the size of the invalid elevation to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        invalid_elevation = cv2.dilate(np.uint8(invalid_elevation) * 255, kernel, iterations=1) == 255
        return invalid_elevation

    def callback(self, msg):
        print("called callback")
        # Convert GridMap to numpy array
        layers = {}
        for layer_name in ["sdf", "elevation"]:
            if layer_name in msg.layers:
                data_list = msg.data[msg.layers.index(layer_name)].data
                layout_info = msg.data[msg.layers.index(layer_name)].layout
                n_cols = layout_info.dim[0].size
                n_rows = layout_info.dim[1].size
                layer = np.reshape(np.array(data_list), (n_rows, n_cols))
                layer = layer[::-1, ::-1].transpose().astype(np.float32)
                layers[layer_name] = layer
            else:
                rospy.logwarn(f"Layer {layer_name} not found in GridMap")
                return False

        try:
            res = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, msg.info.header.stamp, timeout=rospy.Duration(0.01)
            )
        except Exception as e:
            error = str(e)
            rospy.logwarn(f"Couldn't get between odom and base {error}")
            return False

        H, W = layers["sdf"].shape
        rot = res.transform.rotation
        yaw = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_euler("zxy", degrees=False)[0]

        mask_pattern = self.get_pattern_mask(H, W, yaw)
        mask_elevation = self.get_elevation_mask(layers["elevation"])

        for filter_func in self.filter_chain_funcs:
            layers["sdf"] = filter_func(yaw, layers["sdf"])

        layers["sdf"][mask_pattern] = -np.inf
        layers["sdf"][mask_elevation] = -np.inf

        if layers["sdf"].min() == layers["sdf"].max():
            rospy.logwarn(f"No valid elevation within the SDF of the defined pattern {e}")
            return

        # Get index of the maximum gridmax cell index within the SDF
        x, y = np.where(layers["sdf"] == layers["sdf"].max())
        x = x[0]
        y = y[0]

        # Convert the GridMap index to a map frame position
        x -= H / 2
        y -= W / 2
        x *= msg.info.resolution
        y *= msg.info.resolution
        x += msg.info.pose.position.x
        y += msg.info.pose.position.y

        # Publish the goal
        goal = PoseWithCovarianceStamped()
        goal.header.stamp = msg.info.header.stamp
        goal.header.frame_id = self.map_frame
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = res.transform.translation.z + 0.3
        goal.pose.pose.orientation = rot
        self.pub_goal.publish(goal)

        if self.debug:
            # Republish the SDF used to search for the maximum goal
            msg.data[msg.layers.index("sdf")].data = layers["sdf"][::-1, ::-1].transpose().ravel()
            self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("wild_visual_navigation_smart_carrot")
    print("Start")
    wvn = SmartCarrotNode()
    rospy.spin()
