#!/usr/bin/env python
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#

import rospy
from anymal_msgs.msg import AnymalState
from sensor_msgs.msg import JointState
import tf2_ros
import geometry_msgs.msg

# for wvn
from wild_visual_navigation_msgs.msg import CustomState, RobotState

labels_initialized = False
# Preallocate messages
robot_state_msg = RobotState()
# Extract joint states
joint_position = CustomState()
joint_position.name = "joint_position"
joint_position.dim = 12
joint_position.labels = [""] * joint_position.dim
joint_position.values = [0] * joint_position.dim
robot_state_msg.states.append(joint_position)

# Joint velocity
joint_velocity = CustomState()
joint_velocity.name = "joint_velocity"
joint_velocity.dim = 12
joint_velocity.labels = [""] * joint_velocity.dim
joint_velocity.values = [0] * joint_velocity.dim
robot_state_msg.states.append(joint_velocity)

# Acceleration
joint_acceleration = CustomState()
joint_acceleration.name = "joint_acceleration"
joint_acceleration.dim = 12
joint_acceleration.labels = [""] * joint_acceleration.dim
joint_acceleration.values = [0] * joint_acceleration.dim
robot_state_msg.states.append(joint_acceleration)

# Effort
joint_effort = CustomState()
joint_effort.name = "joint_effort"
joint_effort.dim = 12
joint_effort.labels = [""] * joint_effort.dim
joint_effort.values = [0] * joint_effort.dim
robot_state_msg.states.append(joint_effort)

# Vector state
vector_state = CustomState()
vector_state.name = "vector_state"
vector_state.dim = 7 + 6  # + 4 * 12
vector_state.values = [0] * vector_state.dim
vector_state.labels = [""] * vector_state.dim
vector_state.values = [0] * vector_state.dim
robot_state_msg.states.append(vector_state)


i = 0
for x in ["tx", "ty", "tz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"]:
    robot_state_msg.states[4].labels[i] = x
    i += 1


def anymal_state_callback(anymal_state_msg):
    global labels_initialized

    # Joint states for URDF
    joint_state_msg = JointState()
    joint_state_msg.header = anymal_state_msg.joints.header
    joint_state_msg.header.frame_id = ""
    joint_state_msg.name = anymal_state_msg.joints.name
    joint_state_msg.position = anymal_state_msg.joints.position
    # joint_state_msg.velocity = anymal_state_msg.joints.velocity
    # joint_state_msg.effort = anymal_state_msg.joints.effort
    joint_state_publisher.publish(joint_state_msg)
    # Pose for odometry
    pose = anymal_state_msg.pose.pose

    # Create a TransformStamped message
    transform_stamped = geometry_msgs.msg.TransformStamped()
    transform_stamped.header = anymal_state_msg.pose.header
    transform_stamped.child_frame_id = "base"
    transform_stamped.transform.translation = pose.position
    transform_stamped.transform.rotation = pose.orientation

    # Publish the TF transform
    tf_broadcaster.sendTransform(transform_stamped)

    # WVN

    # For RobotState msg
    robot_state_msg.header = anymal_state_msg.header

    # Extract pose
    robot_state_msg.pose = anymal_state_msg.pose

    # Extract twist
    robot_state_msg.twist = anymal_state_msg.twist
    # Vector state
    robot_state_msg.states[4].values[0] = anymal_state_msg.pose.pose.position.x
    robot_state_msg.states[4].values[1] = anymal_state_msg.pose.pose.position.y
    robot_state_msg.states[4].values[2] = anymal_state_msg.pose.pose.position.z
    robot_state_msg.states[4].values[3] = anymal_state_msg.pose.pose.orientation.x
    robot_state_msg.states[4].values[4] = anymal_state_msg.pose.pose.orientation.y
    robot_state_msg.states[4].values[5] = anymal_state_msg.pose.pose.orientation.z
    robot_state_msg.states[4].values[6] = anymal_state_msg.pose.pose.orientation.w
    robot_state_msg.states[4].values[7] = anymal_state_msg.twist.twist.linear.x
    robot_state_msg.states[4].values[8] = anymal_state_msg.twist.twist.linear.y
    robot_state_msg.states[4].values[9] = anymal_state_msg.twist.twist.linear.z
    robot_state_msg.states[4].values[10] = anymal_state_msg.twist.twist.angular.x
    robot_state_msg.states[4].values[11] = anymal_state_msg.twist.twist.angular.y
    robot_state_msg.states[4].values[12] = anymal_state_msg.twist.twist.angular.z
    robot_state_pub.publish(robot_state_msg)


if __name__ == "__main__":
    rospy.init_node("anymal_state_republisher")
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Read parameters
    joint_states_topic = rospy.get_param("~joint_states")
    anymal_state_topic = rospy.get_param(
        "~anymal_state_topic",
    )
    output_topic = rospy.get_param("~output_topic")
    joint_state_publisher = rospy.Publisher(joint_states_topic, JointState, queue_size=10)
    robot_state_pub = rospy.Publisher(output_topic, RobotState, queue_size=10)
    rospy.Subscriber(anymal_state_topic, AnymalState, anymal_state_callback, queue_size=1)
    rospy.spin()
