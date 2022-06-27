from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import TemporalGraph, BaseNode
from wild_visual_navigation_ros.ros_converter import ros_tf_to_torch, ros_image_to_torch, anymal_state_to_torch

from anymal_msgs.msg import AnymalState
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import tf


class RosInterface:
    def __init__(self):
        # Read params
        self.read_params()

        # Initialize traversability estimator
        self.traversability_estimator = TraversabilityEstimator(self.time_window)

    def read_params(self):
        # Topics
        self.anymal_state_topic = rospy.get_param("anymal_state_topic", "/state_estimator/anymal_state")
        self.image_topic = rospy.get_param("anymal_state_topic", "/alphasense_driver_ros/cam4/debayered")
        self.info_topic = rospy.get_param("anymal_state_topic", "/alphasense_driver_ros/cam4/debayered/camera_info")

        # Frames
        self.fixed_frame = rospy.get_param("fixed_frame", "odom")
        self.base_frame = rospy.get_param("base_frame", "base")
        self.cam_frame = rospy.get_param("cam_frame", "/cam4_sensor_frame_helper")
        self.footprint_frame = rospy.get_param("footprint_frame", "footprint")

        # Robot size
        self.robot_width = rospy.get_param("robot_width", 0.4)
        self.robot_length = rospy.get_param("robot_length", 0.6)
        self.robot_height = rospy.get_param("robot_height", 0.3)

        # Time window
        self.time_window = rospy.get_param("time_window", 100)

    def setup_ros(self):
        # Anymal state callback
        self.anymal_state_sub = rospy.Subscriber(self.anymal_state_topic, AnymalState, self.anymal_state_callback)

        # Image callback
        image_sub = message_filters.Subscriber(self.image_topic, Image)
        info_sub = message_filters.Subscriber(self.info_topic, CameraInfo)
        ts = message_filters.ExactTimeSynchronizer([self.image_sub, self.info_sub], 1)
        self.ts.registerCallback(self.image_callback())

        # Services
        # Like, reset graph or the like

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

    def query_tf(self, parent_frame, child_frame):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
        return (trans, rot)

    def anymal_state_callback(self, msg):
        ts = msg.header.stamp.toSec()

        # Query transforms from TF
        T_WB = tf_pose_to_torch(self.query_tf(self.fixed_frame, self.base_frame))
        T_BF = tf_pose_to_torch(self.query_tf(self.base_frame, self.footprint_frame))

        # Convert state to tensor
        proprio_tensor, proprio_labels = anymal_state_to_torch(msg)

        # Create proprioceptive node for the graph
        proprio_node = LocalProprioceptionNode(
            timestamp=ts,
            T_WB=T_WB,
            T_BF=T_BF,
            width=self.robot_width,
            length=self.robot_length,
            height=self.robot_width,
            proprioception=proprio_tensor,
        )
        # Add node to graph
        self.traversability_estimator.add_local_node(proprio_node)

    def image_callback(self, image_msg, info_msg):
        ts = image_msg.header.stamp.toSec()

        # Query transforms from TF
        T_WB = tf_pose_to_torch(self.query_tf(self.fixed_frame, self.base_frame))
        T_BC = tf_pose_to_torch(self.query_tf(self.base_frame, self.camera_frame))

        # Prepare image projector
        image_projector = ImageProjector(torch.FloatTensor(info_msg.K), info_msg.height, info_msg.width)

        # Add image to base node
        # convert image message to torch image
        torch_image = ros_image_to_torch(image_msg)

        # Create image node for the graph
        image_node = LocalImageNode(timestamp=ts, T_WB=T_WB, T_BC=T_BC, image=torch_image, projector=image_projector)

        # Add node to graph
        self.traversability_estimator.add_local_node(image_node)

    def learning_callback(self):
        # Update reprojections
        self.traversability_estimator.update_labels(search_radius=self.traversability_radius)

        # Train traversability
        self.traversability_estimator.train(iter=10)

        # publish reprojections
        for node in self.traversability_estimator.get_nodes():
            torch_mask = node.get_traversability_mask()

        # publish traversability


if __name__ == "__main__":
    rospy.init_node("listener", anonymous=True)

    rospy.spin()
