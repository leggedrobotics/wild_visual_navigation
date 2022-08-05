#!/usr/bin/python3
from wild_visual_navigation_msgs.msg import CustomState, RobotState
from anymal_msgs.msg import AnymalState
import rospy


def anymal_msg_callback(anymal_state):
    # For RobotState msg
    robot_state_msg = RobotState()
    robot_state_msg.header = anymal_state.header

    # Extract pose
    robot_state_msg.pose_frame_id = anymal_state.header.frame_id
    robot_state_msg.pose = anymal_state.pose

    # Extract twist
    robot_state_msg.twist_frame_id = "base"  # TODO this should't be hardcoded
    robot_state_msg.twist = anymal_state.twist

    # Extract joint states
    joint_position = CustomState()
    joint_position.name = "joint_position"
    joint_position.dim = 12
    joint_position.labels = anymal_state.joints.name
    joint_position.values = anymal_state.joints.position
    robot_state_msg.states.append(joint_position)

    joint_velocity = CustomState()
    joint_velocity.name = "joint_velocity"
    joint_velocity.dim = 12
    joint_velocity.labels = anymal_state.joints.name
    joint_velocity.values = anymal_state.joints.position
    robot_state_msg.states.append(joint_velocity)

    joint_acceleration = CustomState()
    joint_acceleration.name = "joint_acceleration"
    joint_acceleration.dim = 12
    joint_acceleration.labels = anymal_state.joints.name
    joint_acceleration.values = anymal_state.joints.position
    robot_state_msg.states.append(joint_acceleration)

    joint_effort = CustomState()
    joint_effort.name = "joint_effort"
    joint_effort.dim = 12
    joint_effort.labels = anymal_state.joints.name
    joint_effort.values = anymal_state.joints.position
    robot_state_msg.states.append(joint_effort)

    vector_state = CustomState()
    vector_state.name = "vector_state"
    vector_state.dim = 7 + 6 + 4 * 12
    vector_state.labels.extend(["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    vector_state.labels.extend(["vx", "vy", "vz", "wx", "wy", "wz"])
    vector_state.labels.extend([f"position_{x}" for x in anymal_state.joints.name])
    vector_state.labels.extend([f"velocity_{x}" for x in anymal_state.joints.name])
    vector_state.labels.extend([f"acceleration_{x}" for x in anymal_state.joints.name])
    vector_state.labels.extend([f"effort_{x}" for x in anymal_state.joints.name])
    vector_state.values.append(anymal_state.pose.pose.position.x)
    vector_state.values.append(anymal_state.pose.pose.position.y)
    vector_state.values.append(anymal_state.pose.pose.position.z)
    vector_state.values.append(anymal_state.pose.pose.orientation.x)
    vector_state.values.append(anymal_state.pose.pose.orientation.y)
    vector_state.values.append(anymal_state.pose.pose.orientation.z)
    vector_state.values.append(anymal_state.pose.pose.orientation.w)
    vector_state.values.append(anymal_state.twist.twist.linear.x)
    vector_state.values.append(anymal_state.twist.twist.linear.y)
    vector_state.values.append(anymal_state.twist.twist.linear.z)
    vector_state.values.append(anymal_state.twist.twist.angular.x)
    vector_state.values.append(anymal_state.twist.twist.angular.y)
    vector_state.values.append(anymal_state.twist.twist.angular.z)
    # Append joint position
    vector_state.values.extend([x for x in anymal_state.joints.position])
    # Append joint velocity
    vector_state.values.extend([x for x in anymal_state.joints.velocity])
    # Append joint acceleration
    vector_state.values.extend([x for x in anymal_state.joints.acceleration])
    # Append joint effort
    vector_state.values.extend([x for x in anymal_state.joints.effort])
    robot_state_msg.states.append(vector_state)

    # Publish
    robot_state_pub.publish(robot_state_msg)


if __name__ == "__main__":
    rospy.init_node("anymal_msg_converter_node")

    # Read parameters
    anymal_state_topic = rospy.get_param("~anymal_state_topic", "/state_estimator/anymal_state")
    output_topic = rospy.get_param("~output_topic", "/wild_visual_navigation_ros/robot_state")

    # Set publishers and subscribers
    anymal_state_sub = rospy.Subscriber(anymal_state_topic, AnymalState, anymal_msg_callback, queue_size=1)
    robot_state_pub = rospy.Publisher(output_topic, RobotState, queue_size=10)

    rospy.loginfo("[anymal_msg_converter_node] ready")
    rospy.spin()
