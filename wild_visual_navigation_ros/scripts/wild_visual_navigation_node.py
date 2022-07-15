#!/usr/bin/python3
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import LocalImageNode, LocalProprioceptionNode
import wild_visual_navigation_ros.ros_converter as rc

from anymal_msgs.msg import AnymalState
from geometry_msgs.msg import Pose, PoseStamped, Point
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import message_filters
import rospy
import tf
import torch

torch.cuda.empty_cache()


class WvnRosInterface:
    def __init__(self):
        # Read params
        self.read_params()

        # Initialize traversability estimator
        self.traversability_estimator = TraversabilityEstimator(
            device=self.device, time_window=self.time_window, image_distance_thr=0.4, proprio_distance_thr=0.4
        )

        # Setup ros
        self.setup_ros()

        rospy.loginfo("[WVN] System ready")
        rospy.spin()

    def read_params(self):
        # Topics
        self.anymal_state_topic = rospy.get_param("~anymal_state_topic", "/state_estimator/anymal_state")
        self.image_topic = rospy.get_param("~anymal_state_topic", "/alphasense_driver_ros/cam4/debayered")
        self.info_topic = rospy.get_param("~anymal_state_topic", "/alphasense_driver_ros/cam4/camera_info")

        # Frames
        self.fixed_frame = rospy.get_param("~fixed_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.camera_frame = rospy.get_param("~camera_frame", "cam4_sensor_frame_helper")
        self.footprint_frame = rospy.get_param("~footprint_frame", "footprint")

        # Robot size
        self.robot_length = rospy.get_param("~robot_length", 1.0)
        self.robot_width = rospy.get_param("~robot_width", 0.6)
        self.robot_height = rospy.get_param("~robot_height", 0.3)

        # Time window
        self.time_window = rospy.get_param("~time_window", 10)
        self.learning_timer_freq = rospy.get_param("~learning_timer_freq", 0.2)  # hertz
        self.vis_timer_freq = rospy.get_param("~visualization_timer_freq", 1)  # hertz

        # Traversability estimation params
        self.traversability_radius = rospy.get_param("~traversability_radius", 5.0)

        # Torch device
        self.device = rospy.get_param("device", "cuda")

    def setup_ros(self):
        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # Anymal state callback
        self.anymal_state_sub = rospy.Subscriber(
            self.anymal_state_topic, AnymalState, self.anymal_state_callback, queue_size=1
        )

        # Image callback
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.info_sub = message_filters.Subscriber(self.info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub], 1, slop=1.0 / 10)
        self.ts.registerCallback(self.image_callback)

        # Publishers
        self.pub_debug_image_labeled = rospy.Publisher(
            "/wild_visual_navigation_node/debug/last_node_image_labeled", Image, queue_size=10
        )
        self.pub_debug_image_mask = rospy.Publisher(
            "/wild_visual_navigation_node/debug/last_node_image_mask", Image, queue_size=10
        )
        self.pub_debug_image_features = rospy.Publisher(
            "/wild_visual_navigation_node/debug/last_node_image_features", Image, queue_size=10
        )
        self.pub_debug_local_proprio_graph = rospy.Publisher(
            "/wild_visual_navigation_node/debug/local_proprioceptive_graph", Path, queue_size=10
        )
        self.pub_debug_local_image_graph = rospy.Publisher(
            "/wild_visual_navigation_node/debug/local_image_graph", Path, queue_size=10
        )
        self.pub_debug_local_graph_footprints = rospy.Publisher(
            "/wild_visual_navigation_node/debug/local_graph_footprints", Marker, queue_size=10
        )

        # Services
        # Like, reset graph or the like

    def query_tf(self, parent_frame, child_frame):
        self.tf_listener.waitForTransform(parent_frame, child_frame, rospy.Time(), rospy.Duration(1.0))

        try:
            (trans, rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            return (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)

    def anymal_state_callback(self, msg):
        ts = msg.header.stamp.to_sec()

        # Query transforms from TF
        T_WB = rc.ros_tf_to_torch(self.query_tf(self.fixed_frame, self.base_frame), device=self.device)
        T_BF = rc.ros_tf_to_torch(self.query_tf(self.base_frame, self.footprint_frame), device=self.device)

        # The footprint requires a correction: we use the same orientation as the base
        T_BF[:3, :3] = torch.eye(3, device=self.device)

        # Convert state to tensor
        proprio_tensor, proprio_labels = rc.anymal_state_to_torch(msg, device=self.device)

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
        self.traversability_estimator.add_local_proprio_node(proprio_node)

    def image_callback(self, image_msg, info_msg):
        ts = image_msg.header.stamp.to_sec()

        # Query transforms from TF
        T_WB = rc.ros_tf_to_torch(self.query_tf(self.fixed_frame, self.base_frame), device=self.device)
        T_BC = rc.ros_tf_to_torch(self.query_tf(self.base_frame, self.camera_frame), device=self.device)

        # Prepare image projector
        K, H, W = rc.ros_cam_info_to_tensors(info_msg, device=self.device)
        image_projector = ImageProjector(K, H, W)

        # Add image to base node
        # convert image message to torch image
        torch_image = rc.ros_image_to_torch(image_msg, device=self.device)

        # Create image node for the graph
        image_node = LocalImageNode(timestamp=ts, T_WB=T_WB, T_BC=T_BC, image=torch_image, projector=image_projector)

        # Add node to graph
        self.traversability_estimator.add_local_image_node(image_node)

        self.learn(None)
        self.visualize(None)

    def learn(self, event):
        # Update reprojections
        self.traversability_estimator.update_labels_and_features(search_radius=self.traversability_radius)

        # Train traversability
        self.traversability_estimator.train(iter=10)

        # publish traversability

    def visualize(self, event):
        now = rospy.Time.now()
        # publish reprojections of last node in graph
        if len(self.traversability_estimator.get_local_debug_nodes()) > 0:
            last_node = self.traversability_estimator.get_local_debug_nodes()[0]

            ros_mask = last_node.get_traversability_mask()
            ros_labeled_image = last_node.get_labeled_image()
            ros_features_image = last_node.get_features_image()
            if ros_mask is not None:
                self.pub_debug_image_mask.publish(rc.torch_to_ros_image(ros_mask))
            if ros_labeled_image is not None:
                self.pub_debug_image_labeled.publish(rc.torch_to_ros_image(ros_labeled_image))
            if ros_features_image is not None:
                self.pub_debug_image_features.publish(rc.torch_to_ros_image(ros_features_image))

        # Publish local graph
        local_proprio_graph_msg = Path()
        local_proprio_graph_msg.header.frame_id = self.fixed_frame
        local_proprio_graph_msg.header.stamp = now

        local_image_graph_msg = Path()
        local_image_graph_msg.header.frame_id = self.fixed_frame
        local_image_graph_msg.header.stamp = now
        # Footprints
        footprints_marker = Marker()
        footprints_marker.id = 0
        footprints_marker.ns = "footprints"
        footprints_marker.header.frame_id = self.fixed_frame
        footprints_marker.header.stamp = now
        footprints_marker.type = Marker.TRIANGLE_LIST
        footprints_marker.action = Marker.ADD
        footprints_marker.scale.x = 1
        footprints_marker.scale.y = 1
        footprints_marker.scale.z = 1
        footprints_marker.color.a = 0.5
        footprints_marker.color.r = 0.0
        footprints_marker.color.g = 1.0
        footprints_marker.color.b = 0.0
        footprints_marker.pose.orientation.w = 1.0
        footprints_marker.pose.position.x = 0.0
        footprints_marker.pose.position.y = 0.0
        footprints_marker.pose.position.z = 0.0

        for node in self.traversability_estimator.get_local_proprio_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.get_pose_base_in_world())
            local_proprio_graph_msg.poses.append(pose)

            # Footprints
            footprint_points = node.get_footprint_points().unsqueeze(0)
            B, N, D = footprint_points.shape
            for n in [0, 2, 1, 2, 0, 3]:  # this is a hack to show the triangles correctly
                p = Point()
                p.x = footprint_points[0, n, 0]
                p.y = footprint_points[0, n, 1]
                p.z = footprint_points[0, n, 2]
                footprints_marker.points.append(p)

        for node in self.traversability_estimator.get_local_image_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.get_pose_cam_in_world())
            local_image_graph_msg.poses.append(pose)

        self.pub_debug_local_proprio_graph.publish(local_proprio_graph_msg)
        self.pub_debug_local_image_graph.publish(local_image_graph_msg)
        self.pub_debug_local_graph_footprints.publish(footprints_marker)


if __name__ == "__main__":
    rospy.init_node("wild_visual_navigation_node")
    wvn = WvnRosInterface()
