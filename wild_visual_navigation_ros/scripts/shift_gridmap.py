from grid_map_msgs.msg import GridMap, GridMapInfo
import rospy
import sys


class ShiftGridMapNode:
    def __init__(self):
        self.gridmap_sub_topic = rospy.get_param("~gridmap_sub_topic", "/ele_trash/elevation_map_wifi")
        print("Subscribed", self.gridmap_sub_topic)
        self.gridmap_pub_topic = rospy.get_param("~gridmap_pub_topic", "elevation_map_wifi_shifted")
        self.offset_factor_x = rospy.get_param("~offset_factor_x")
        self.offset_factor_y = rospy.get_param("~offset_factor_y")

        self.pub = rospy.Publisher(f"~{self.gridmap_pub_topic}", GridMap, queue_size=10)
        rospy.Subscriber(f"~{self.gridmap_sub_topic}", GridMap, self.callback, queue_size=1)

    def callback(self, msg):
        msg.info.pose.position.x += msg.info.length_x * self.offset_factor_x
        msg.info.pose.position.y += msg.info.length_y * self.offset_factor_y
        self.pub.publish(msg)


if __name__ == "__main__":
    print("MAIN NAME")
    try:
        nr = "_" + rospy.myargv(argv=sys.argv)[-1].split(" ")[-1]
        rospy.init_node(f"wild_visual_navigation_gridmap_{nr}")
    except:
        rospy.init_node("wild_visual_navigation_gridmap")
    print("Start")
    wvn = ShiftGridMapNode()
    rospy.spin()
