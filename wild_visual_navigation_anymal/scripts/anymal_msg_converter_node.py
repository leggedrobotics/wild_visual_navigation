#!/usr/bin/python3
from wild_visual_navigation import WVN_ROOT_DIR
from anymal_msgs.msg import AnymalState
from wild_visual_navigation_msgs.msg import CustomState, RobotState


if __name__ == "__main__":
    rospy.init_node("anymal_msg_converter_node")
    # TODO
    rospy.spin()
