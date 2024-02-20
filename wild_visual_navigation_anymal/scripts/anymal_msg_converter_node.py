#!/usr/bin/python3
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation_msgs.msg import CustomState, RobotState
from anymal_msgs.msg import AnymalState
import rospy

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


def anymal_msg_callback(anymal_state, return_msg=False):
    global labels_initialized

    # For RobotState msg
    robot_state_msg.header = anymal_state.header

    # Extract pose
    robot_state_msg.pose = anymal_state.pose

    # Extract twist
    robot_state_msg.twist = anymal_state.twist

    # # Joints
    # # Joint position
    # robot_state_msg.states[0].labels = anymal_state.joints.name
    # robot_state_msg.states[0].values = anymal_state.joints.position
    # # Joint velocity
    # robot_state_msg.states[1].labels = anymal_state.joints.name
    # robot_state_msg.states[1].values = anymal_state.joints.velocity
    # # Joint acceleration
    # robot_state_msg.states[2].labels = anymal_state.joints.name
    # robot_state_msg.states[2].values = anymal_state.joints.acceleration
    # # Joint effort
    # robot_state_msg.states[3].labels = anymal_state.joints.name
    # robot_state_msg.states[3].values = anymal_state.joints.effort

    # Vector state
    robot_state_msg.states[4].values[0] = anymal_state.pose.pose.position.x
    robot_state_msg.states[4].values[1] = anymal_state.pose.pose.position.y
    robot_state_msg.states[4].values[2] = anymal_state.pose.pose.position.z
    robot_state_msg.states[4].values[3] = anymal_state.pose.pose.orientation.x
    robot_state_msg.states[4].values[4] = anymal_state.pose.pose.orientation.y
    robot_state_msg.states[4].values[5] = anymal_state.pose.pose.orientation.z
    robot_state_msg.states[4].values[6] = anymal_state.pose.pose.orientation.w
    robot_state_msg.states[4].values[7] = anymal_state.twist.twist.linear.x
    robot_state_msg.states[4].values[8] = anymal_state.twist.twist.linear.y
    robot_state_msg.states[4].values[9] = anymal_state.twist.twist.linear.z
    robot_state_msg.states[4].values[10] = anymal_state.twist.twist.angular.x
    robot_state_msg.states[4].values[11] = anymal_state.twist.twist.angular.y
    robot_state_msg.states[4].values[12] = anymal_state.twist.twist.angular.z

    # i = 13
    # # Joints
    # if not labels_initialized:
    #     for l in anymal_state.joints.name:
    #         robot_state_msg.states[4].labels[i] = f"position_{l}"
    #         robot_state_msg.states[4].labels[i + 12] = f"velocity_{l}"
    #         robot_state_msg.states[4].labels[i + 2*12] = f"acceleration_{l}"
    #         robot_state_msg.states[4].labels[i + 3*12] = f"effort_{l}"
    #         i += 1
    #     labels_initialized = True

    # # Assign values
    # d = 4*12
    # i = 13
    # robot_state_msg.states[4].values[i:i+d] = anymal_state.joints.position
    # i = i + d
    # robot_state_msg.states[4].values[i:i+d] = anymal_state.joints.velocity
    # i = i + d
    # robot_state_msg.states[4].values[i:i+d] = anymal_state.joints.acceleration
    # i = i + d
    # robot_state_msg.states[4].values[i:i+d] = anymal_state.joints.effort

    if return_msg:
        return robot_state_msg
    # Publish
    robot_state_pub.publish(robot_state_msg)


if __name__ == "__main__":
    rospy.init_node("anymal_msg_converter_node")

    # Read parameters
    anymal_state_topic = rospy.get_param("~anymal_state_topic", "/state_estimator/anymal_state")
    output_topic = rospy.get_param("~output_topic", "/wild_visual_navigation_node/robot_state")

    # Set publishers and subscribers
    robot_state_pub = rospy.Publisher(output_topic, RobotState, queue_size=20)
    anymal_state_sub = rospy.Subscriber(anymal_state_topic, AnymalState, anymal_msg_callback, queue_size=20)

    rospy.loginfo("[anymal_msg_converter_node] ready")
    rospy.spin()
