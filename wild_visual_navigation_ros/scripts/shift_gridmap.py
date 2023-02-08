from grid_map_msgs.msg import GridMap, GridMapInfo
import rospy
import sys
import tf
from tf.transformations import euler_from_quaternion
import math
import numpy as np


class ShiftGridMapNode:
    def __init__(self):
        self.gridmap_sub_topic = rospy.get_param("~gridmap_sub_topic", "/recorded/elevation_map_wifi")
        print("Subscribed", self.gridmap_sub_topic)
        self.gridmap_pub_topic = rospy.get_param("~gridmap_pub_topic", "elevation_map_wifi_shifted")
        self.offset_factor_x = rospy.get_param("~offset_factor_x")
        self.offset_factor_y = rospy.get_param("~offset_factor_y")
        self.compensate_tf = rospy.get_param("~compensate_tf", False)

        if self.compensate_tf:
            self.listener = tf.TransformListener()

        self.pub = rospy.Publisher(f"~{self.gridmap_pub_topic}", GridMap, queue_size=10)
        rospy.Subscriber(f"~{self.gridmap_sub_topic}", GridMap, self.callback, queue_size=1)

    def callback(self, msg):
        delta_x = msg.info.length_x * self.offset_factor_x
        delta_y = msg.info.length_y * self.offset_factor_y

        if self.compensate_tf:
            try:
                (trans, rot) = self.listener.lookupTransform(msg.info.header.frame_id, "base", msg.info.header.stamp)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return
            yaw = euler_from_quaternion(rot)[2]
        else:
            yaw = 0

        msg.info.pose.position.x += delta_x * math.cos(yaw) - delta_y * math.sin(yaw)
        msg.info.pose.position.y += delta_x * math.sin(yaw) + delta_y * math.cos(yaw)
        self.pub.publish(msg)


if __name__ == "__main__":
    print("Starting Shift GridMap Node!")
    # rosrun wild_visual_navigation_ros shift_gridmap.py 0 _gridmap_sub_topic:=/elevation_mapping/elevation_map_wifi _gridmap_pub_topic:=/left _offset_factor_x:=10 _offset_factor_y:=0

    try:
        nr = "_" + rospy.myargv(argv=sys.argv)[-1].split(" ")[-1]
        rospy.init_node(f"wild_visual_navigation_gridmap_{nr}")
    except:
        rospy.init_node("wild_visual_navigation_gridmap")

    wvn = ShiftGridMapNode()
    rospy.spin()

    print("Finished Shift GridMap Node!")
