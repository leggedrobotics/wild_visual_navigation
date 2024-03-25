#!/usr/bin/python3
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from wild_visual_navigation_msgs.msg import RobotState, CustomState
import rospy


def jackal_msg_callback(jackal_state, return_msg=False):
    robot_state_msg = RobotState()

    # For RobotState msg
    robot_state_msg.header = jackal_state.header

    # Extract pose
    robot_state_msg.pose.header = jackal_state.header
    robot_state_msg.pose.pose = jackal_state.pose.pose

    # Extract twist
    robot_state_msg.twist.header = jackal_state.header
    robot_state_msg.twist.header.frame_id = jackal_state.child_frame_id
    robot_state_msg.twist.twist = jackal_state.twist.twist

    vector_state = CustomState()
    vector_state.name = "vector_state"
    vector_state.dim = 7 + 6  # + 4 * 12
    vector_state.values = [0] * vector_state.dim
    vector_state.labels = [""] * vector_state.dim
    vector_state.values = [0] * vector_state.dim
    robot_state_msg.states.append(vector_state)

    robot_state_msg.states[0].values[0] = robot_state_msg.pose.pose.position.x
    robot_state_msg.states[0].values[1] = robot_state_msg.pose.pose.position.y
    robot_state_msg.states[0].values[2] = robot_state_msg.pose.pose.position.z
    robot_state_msg.states[0].values[3] = robot_state_msg.pose.pose.orientation.x
    robot_state_msg.states[0].values[4] = robot_state_msg.pose.pose.orientation.y
    robot_state_msg.states[0].values[5] = robot_state_msg.pose.pose.orientation.z
    robot_state_msg.states[0].values[6] = robot_state_msg.pose.pose.orientation.w
    robot_state_msg.states[0].values[7] = robot_state_msg.twist.twist.linear.x
    robot_state_msg.states[0].values[8] = robot_state_msg.twist.twist.linear.y
    robot_state_msg.states[0].values[9] = robot_state_msg.twist.twist.linear.z
    robot_state_msg.states[0].values[10] = robot_state_msg.twist.twist.angular.x
    robot_state_msg.states[0].values[11] = robot_state_msg.twist.twist.angular.y
    robot_state_msg.states[0].values[12] = robot_state_msg.twist.twist.angular.z

    for i, x in enumerate(["tx", "ty", "tz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"]):
        robot_state_msg.states[0].labels[i] = x

    if return_msg:
        return robot_state_msg
    # Publish
    robot_state_pub.publish(robot_state_msg)


def twist_msg_callback(msg):
    ts = rospy.Time.now()
    out_msg = TwistStamped()
    out_msg.header.stamp = ts
    out_msg.header.frame_id = "base_link"
    out_msg.twist = msg

    ref_twiststamped_pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("jackal_state_converter_node")

    # We subscribe the odometry topic (state)
    jackal_state_sub = rospy.Subscriber("/odometry/filtered", Odometry, jackal_msg_callback, queue_size=20)
    robot_state_pub = rospy.Publisher("/wild_visual_navigation_node/robot_state", RobotState, queue_size=20)

    # And also the twist command from teleoperation
    ref_twist_sub = rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, twist_msg_callback, queue_size=20)
    ref_twiststamped_pub = rospy.Publisher("/wild_visual_navigation_node/reference_twist", TwistStamped, queue_size=20)

    rospy.loginfo("[jackal_state_converter_node] ready")
    rospy.spin()
