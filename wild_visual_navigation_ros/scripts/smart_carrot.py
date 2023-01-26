from grid_map_msgs.msg import GridMap, GridMapInfo
import rospy
import sys
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
import cv2
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
import time


class SmartCarrotNode:
    def __init__(self):
        self.gridmap_sub_topic = "/elevation_mapping/elevation_map_wifi"  # rospy.get_param("~gridmap_sub_topic", "/elevation_mapping/elevation_map_wifi")
        self.pub = rospy.Publisher(f"~binary_mask", GridMap, queue_size=5)
        self.pub_goal = rospy.Publisher(f"/initialpose", PoseWithCovarianceStamped, queue_size=5)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber(f"~{self.gridmap_sub_topic}", GridMap, self.callback, queue_size=10)

        self.offset = np.zeros((200, 200))
        for x in range(200):
            for y in range(200):
                self.offset[x, y] = math.sqrt((x - 100) ** 2 + (y - 100) ** 2)

        self.offset /= self.offset.max()
        self.offset *= 0.3
        self.debug = False

    def callback(self, msg):
        st = time.time()
        target_layer = "sdf"
        if target_layer in msg.layers:
            # extract grid_map layer as numpy array
            data_list = msg.data[msg.layers.index(target_layer)].data
            layout_info = msg.data[msg.layers.index(target_layer)].layout
            n_cols = layout_info.dim[0].size
            n_rows = layout_info.dim[1].size
            sdf = np.reshape(np.array(data_list), (n_rows, n_cols))
            sdf = sdf[::-1, ::-1].transpose().astype(np.float32)

        try:
            res = self.tf_buffer.lookup_transform(
                "odom", "base_inverted_field_local_planner", msg.info.header.stamp, timeout=rospy.Duration(0.01)
            )
        except Exception as e:
            print("error")
            print("Error in query tf: ", e)
            rospy.logwarn(f"Couldn't get between odom and base")
            return

        yaw = R.from_quat(
            [res.transform.rotation.x, res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w]
        ).as_euler("zxy", degrees=False)[0]

        binary_mask = np.zeros((sdf.shape[0], sdf.shape[1]), dtype=np.uint8)

        distance = sdf.shape[0] / 5
        center_x = int(sdf.shape[0] / 2 + math.sin(yaw) * distance)
        center_y = int(sdf.shape[1] / 2 + math.cos(yaw) * distance)
        binary_mask = cv2.circle(binary_mask, (center_x, center_y), 0, 255, 30)
        distance = sdf.shape[0] / 3
        center_x = int(sdf.shape[0] / 2 + math.sin(yaw) * distance)
        center_y = int(sdf.shape[1] / 2 + math.cos(yaw) * distance)
        binary_mask = cv2.circle(binary_mask, (center_x, center_y), 0, 255, 50)
        distance = sdf.shape[0] / 2
        center_x = int(sdf.shape[0] / 2 + math.sin(yaw) * distance)
        center_y = int(sdf.shape[1] / 2 + math.cos(yaw) * distance)
        binary_mask = cv2.circle(binary_mask, (center_x, center_y), 0, 255, 100)
        m = binary_mask == 0
        sdf += self.offset
        sdf[m] = sdf.min()

        x, y = np.where(sdf == sdf.max())
        x = x[0]
        y = y[0]
        x -= sdf.shape[0] / 2
        y -= sdf.shape[1] / 2
        x *= msg.info.resolution
        y *= msg.info.resolution
        x += msg.info.pose.position.x
        y += msg.info.pose.position.y

        goal = PoseWithCovarianceStamped()
        goal.header.stamp = msg.info.header.stamp
        goal.header.frame_id = "odom"
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = res.transform.translation.z + 0.3
        goal.pose.pose.orientation = res.transform.rotation
        self.pub_goal.publish(goal)

        if self.debug:
            msg.data[msg.layers.index(target_layer)].data = sdf[::-1, ::-1].transpose().ravel()
            self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("wild_visual_navigation_smart_carrot")
    print("Start")
    wvn = SmartCarrotNode()
    rospy.spin()
