#!/usr/bin/python3
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
# This script implements a simple carrot follower scheme.
# Given a 2D Nav Goal from Rviz, it implements a simple P-control law
# in x, y and yaw to reach the goal.
# Control is done in the "world" frame (see gazebo_world_publisher)
# so there is no state estimation drift for simplicity

from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.msg import LinkStates
import rospy
import math
import numpy as np
import tf.transformations as tr


goal_x, goal_y = (None, None)

MAX_LINEAR_VEL = 0.5
MAX_ANGULAR_VEL = 0.7
GAIN_LINEAR = 1.0
GAIN_ANGULAR = 1.5
DIST_THR = 0.1  # 10 cms


def wrap_angle(angle):
    angle = np.fmod(angle + np.pi, 2 * np.pi)
    if angle < 0:
        angle = angle + 2 * np.pi
    return angle - np.pi


def msg_to_se2(pose):
    x = pose.position.x
    y = pose.position.y
    eul = tr.euler_from_quaternion(
        np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    )
    yaw = eul[2]
    return x, y, yaw


def goal_callback(goal_msg):
    rospy.loginfo("New goal received")
    global goal_x, goal_y
    goal_x, goal_y, _ = msg_to_se2(goal_msg.pose)


def gazebo_callback(msg):
    global goal_x, goal_y
    robot_x, robot_y, robot_yaw = msg_to_se2(msg.pose[1])
    if goal_x is None or goal_y is None:
        return
    compute_cmd(goal_x, goal_y, robot_x, robot_y, robot_yaw)


def compute_cmd(goal_x, goal_y, robot_x, robot_y, robot_yaw):
    if goal_x is None or goal_y is None:
        return

    # Get angular difference
    yaw_diff = wrap_angle(math.atan2(goal_y - robot_y, goal_x - robot_x) - robot_yaw)
    # Get distance to goal
    dist_diff = math.sqrt(((goal_y - robot_y) ** 2) + (goal_x - robot_x) ** 2)

    twist_cmd = Twist()
    if dist_diff > DIST_THR:
        twist_cmd.angular.z = np.clip(GAIN_ANGULAR * yaw_diff, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        twist_cmd.linear.x = np.clip(GAIN_LINEAR * dist_diff, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    else:
        twist_cmd.angular.z = 0.0
        twist_cmd.linear.x = 0.0

    # Compute control law
    cmd_pub.publish(twist_cmd)


if __name__ == "__main__":
    rospy.init_node("jackal_carrot_follower")
    gazebo_sub = rospy.Subscriber("/gazebo/link_states/", LinkStates, gazebo_callback, queue_size=5)
    goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_callback, queue_size=5)

    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
    rospy.loginfo("[jackal_carrot_follower] ready")
    rospy.spin()
