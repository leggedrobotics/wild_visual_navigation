#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray

desired_twist_msg = TwistStamped()


def policy_debug_info_callback(debug_info_msg: Float32MultiArray):
    desired_twist_msg.twist.linear.x = debug_info_msg.data[0]
    desired_twist_msg.twist.linear.y = debug_info_msg.data[1]
    desired_twist_msg.twist.angular.z = debug_info_msg.data[2]
    desired_twist_msg.header.stamp = rospy.Time.now()
    pub.publish(desired_twist_msg)


if __name__ == "__main__":
    rospy.init_node("policy_debug_info_converter_node")
    # Set publishers and subscribers
    sub = rospy.Subscriber(
        "/debug_info",
        Float32MultiArray,
        queue_size=10,
        callback=policy_debug_info_callback,
    )
    pub = rospy.Publisher("/log/state/desiredRobotTwist", TwistStamped, queue_size=10)
    rospy.loginfo("[policy_debug_info_converter_node] ready")
    rospy.spin()
