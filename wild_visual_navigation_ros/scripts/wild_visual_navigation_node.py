#!/usr/bin/python3
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import MissionNode, ProprioceptionNode
import wild_visual_navigation_ros.ros_converter as rc

from wild_visual_navigation_msgs.msg import RobotState
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger, TriggerResponse
from threading import Thread
from visualization_msgs.msg import MarkerArray, Marker
import message_filters
import os
import rospy
import tf
import torch

torch.cuda.empty_cache()


class WvnRosInterface:
    def __init__(self):
        self.last_ts = rospy.get_time()

        # Read params
        self.read_params()

        # Initialize traversability estimator
        self.traversability_estimator = TraversabilityEstimator(
            device=self.device,
            max_distance=self.traversability_radius,
            image_distance_thr=self.image_graph_dist_thr,
            proprio_distance_thr=self.proprio_graph_dist_thr,
        )

        # Setup ros
        self.setup_ros()

        # Launch processes
        print("─" * 80)

        # Setup slow threads
        print("Launching [learning] thread")
        if self.run_online_learning:
            self.learning_thread = Thread(target=self.learning_thread_loop, name="learning")
            self.learning_thread.start()
        print("[WVN] System ready")
        rospy.spin()

    def __del__(self):
        # Join threads
        if self.run_online_learning:
            self.learning_thread.join()

    def read_params(self):
        # Topics
        self.robot_state_topic = rospy.get_param("~robot_state_topic", "/wild_visual_navigation_ros/robot_state")
        self.image_topic = rospy.get_param("~image_topic", "/alphasense_driver_ros/cam4/debayered")
        self.info_topic = rospy.get_param("~camera_info_topic", "/alphasense_driver_ros/cam4/camera_info")

        # Frames
        self.fixed_frame = rospy.get_param("~fixed_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.camera_frame = rospy.get_param("~camera_frame", "cam4_sensor_frame_helper")
        self.footprint_frame = rospy.get_param("~footprint_frame", "footprint")

        # Robot size
        self.robot_length = rospy.get_param("~robot_length", 1.0)
        self.robot_width = rospy.get_param("~robot_width", 0.6)
        self.robot_height = rospy.get_param("~robot_height", 0.3)

        # Traversability estimation params
        self.traversability_radius = rospy.get_param("~traversability_radius", 5.0)
        self.image_graph_dist_thr = rospy.get_param("~image_graph_dist_thr", 0.5)
        self.proprio_graph_dist_thr = rospy.get_param("~proprio_graph_dist_thr", 0.1)
        self.network_input_image_size = rospy.get_param("~network_input_image_size", 448)

        # Threads
        self.run_online_learning = rospy.get_param("~run_online_learning", False)
        self.image_callback_rate = rospy.get_param("~image_callback_rate", 3)  # hertz
        self.learning_thread_rate = rospy.get_param("~learning_thread_rate", 10)  # hertz

        # Data storage
        out_path = os.path.join(WVN_ROOT_DIR, "results")
        self.output_path = rospy.get_param("~output_path", out_path)
        self.mission_name = rospy.get_param(
            "~mission_name", "default_mission"
        )  # Note: We may want to send this in the service call

        # Torch device
        self.device = rospy.get_param("device", "cuda")

    def setup_ros(self):
        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # Robot state callback
        self.robot_state_sub = rospy.Subscriber(
            self.robot_state_topic, RobotState, self.robot_state_callback, queue_size=1
        )

        # Image callback
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.info_sub = message_filters.Subscriber(self.info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub], 1, slop=1.0 / 10)
        self.ts.registerCallback(self.image_callback)

        # Publishers
        self.pub_image_labeled = rospy.Publisher(
            "/wild_visual_navigation_node/last_node_image_labeled", Image, queue_size=10
        )
        self.pub_image_mask = rospy.Publisher("/wild_visual_navigation_node/last_node_image_mask", Image, queue_size=10)
        self.pub_image_prediction = rospy.Publisher(
            "/wild_visual_navigation_node/current_prediction", Image, queue_size=10
        )
        self.pub_image_prediction_uncertainty = rospy.Publisher(
            "/wild_visual_navigation_node/current_uncertainty", Image, queue_size=10
        )
        self.pub_debug_proprio_graph = rospy.Publisher(
            "/wild_visual_navigation_node/proprioceptive_graph", Path, queue_size=10
        )
        self.pub_mission_graph = rospy.Publisher("/wild_visual_navigation_node/mission_graph", Path, queue_size=10)
        self.pub_graph_footprints = rospy.Publisher(
            "/wild_visual_navigation_node/graph_footprints", Marker, queue_size=10
        )

        # Services
        # Like, reset graph or the like
        self.save_graph_service = rospy.Service("~save_graph", Trigger, self.save_graph_callback)
        self.save_pickle_service = rospy.Service("~save_pickle", Trigger, self.save_pickle_callback)

    def save_graph_callback(self, req):
        mission_path = os.path.join(self.output_path, self.mission_name)
        t = Thread(target=self.traversability_estimator.save_graph, args=(mission_path,))
        t.start()
        t.join()
        return TriggerResponse(success=True, message=f"Graph saved in {mission_path}")

    def save_pickle_callback(self, req):
        mission_path = os.path.join(self.output_path, self.mission_name, "traversability_estimator.pickle")
        self.traversability_estimator.save(mission_path)
        return TriggerResponse(success=True, message=f"Pickle saved in {mission_path}")

    def query_tf(self, parent_frame, child_frame):
        self.tf_listener.waitForTransform(parent_frame, child_frame, rospy.Time(), rospy.Duration(1.0))

        try:
            (trans, rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            return (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)

    def robot_state_callback(self, msg):
        ts = msg.header.stamp.to_sec()

        # Query transforms from TF
        pose_base_in_world = rc.ros_tf_to_torch(self.query_tf(self.fixed_frame, self.base_frame), device=self.device)
        pose_footprint_in_base = rc.ros_tf_to_torch(
            self.query_tf(self.base_frame, self.footprint_frame), device=self.device
        )

        # The footprint requires a correction: we use the same orientation as the base
        pose_footprint_in_base[:3, :3] = torch.eye(3, device=self.device)

        # Convert state to tensor
        proprio_tensor, proprio_labels = rc.wvn_robot_state_to_torch(msg, device=self.device)

        # Create proprioceptive node for the graph
        proprio_node = ProprioceptionNode(
            timestamp=ts,
            pose_base_in_world=pose_base_in_world,
            pose_footprint_in_base=pose_footprint_in_base,
            width=self.robot_width,
            length=self.robot_length,
            height=self.robot_width,
            proprioception=proprio_tensor,
        )
        # Add node to graph
        self.traversability_estimator.add_proprio_node(proprio_node)

        # Visualizations
        self.visualize_proprioception()

    def image_callback(self, image_msg, info_msg):
        # Run the callback so as to match the desired rate
        ts = image_msg.header.stamp.to_sec()
        if abs(ts - self.last_ts) < 1.0 / self.image_callback_rate:
            return
        self.last_ts = ts

        # Query transforms from TF
        pose_base_in_world = rc.ros_tf_to_torch(self.query_tf(self.fixed_frame, self.base_frame), device=self.device)
        pose_cam_in_base = rc.ros_tf_to_torch(self.query_tf(self.base_frame, self.camera_frame), device=self.device)

        # Prepare image projector
        K, H, W = rc.ros_cam_info_to_tensors(info_msg, device=self.device)
        image_projector = ImageProjector(K=K, h=H, w=W, new_h=self.network_input_image_size)

        # Add image to base node
        # convert image message to torch image
        torch_image = rc.ros_image_to_torch(image_msg, device=self.device)
        torch_image = image_projector.resize_image(torch_image)

        # Create image node for the graph
        mission_node = MissionNode(
            timestamp=ts,
            pose_base_in_world=pose_base_in_world,
            pose_cam_in_base=pose_cam_in_base,
            image=torch_image,
            image_projector=image_projector,
        )

        # Add node to graph
        self.traversability_estimator.add_mission_node(mission_node)

        # Update prediction for current image
        self.traversability_estimator.update_prediction(mission_node)

        # rospy.loginfo("[main thread] update visualizations")
        self.visualize_mission(mission_node)

    def learning_thread_loop(self):
        # Set rate
        rate = rospy.Rate(self.learning_thread_rate)

        # Main loop
        while not rospy.is_shutdown():
            # Optimize model
            self.traversability_estimator.train()
            rate.sleep()

    def visualize_proprioception(self):
        # Get current time for later
        now = rospy.Time.now()

        proprio_graph_msg = Path()
        proprio_graph_msg.header.frame_id = self.fixed_frame
        proprio_graph_msg.header.stamp = now

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

        for node in self.traversability_estimator.get_proprio_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_base_in_world)
            proprio_graph_msg.poses.append(pose)

            # Footprints
            footprint_points = node.get_footprint_points()[None]
            B, N, D = footprint_points.shape
            for n in [0, 2, 1, 2, 0, 3]:  # this is a hack to show the triangles correctly
                p = Point()
                p.x = footprint_points[0, n, 0]
                p.y = footprint_points[0, n, 1]
                p.z = footprint_points[0, n, 2]
                footprints_marker.points.append(p)

        # Publish
        self.pub_graph_footprints.publish(footprints_marker)
        self.pub_debug_proprio_graph.publish(proprio_graph_msg)

    def visualize_mission(self, mission_node: MissionNode = None):
        # Get current time for later
        now = rospy.Time.now()

        # Publish predictions
        if mission_node is not None:
            torch_prediction_image, torch_uncertainty_image = mission_node.get_prediction_image()
            self.pub_image_prediction.publish(rc.torch_to_ros_image(torch_prediction_image))
            self.pub_image_prediction_uncertainty.publish(rc.torch_to_ros_image(torch_uncertainty_image))

        # Publish reprojections of last node in graph
        # TODO: change visualization for a better node
        if len(self.traversability_estimator.get_mission_nodes()) > 0:
            mission_node = self.traversability_estimator.get_last_valid_mission_node()

            if mission_node is not None:
                torch_mask = mission_node.supervision_mask
                torch_labeled_image = mission_node.get_training_image()
                self.pub_image_mask.publish(rc.torch_to_ros_image(torch_mask))
                self.pub_image_labeled.publish(rc.torch_to_ros_image(torch_labeled_image))

        # Publish local graph
        mission_graph_msg = Path()
        mission_graph_msg.header.frame_id = self.fixed_frame
        mission_graph_msg.header.stamp = now

        for node in self.traversability_estimator.get_mission_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_cam_in_world)
            mission_graph_msg.poses.append(pose)

        self.pub_mission_graph.publish(mission_graph_msg)


if __name__ == "__main__":
    rospy.init_node("wild_visual_navigation_node")
    wvn = WvnRosInterface()
