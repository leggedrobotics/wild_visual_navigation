#!/usr/bin/python3
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.supervision_generator import SupervisionGenerator
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import MissionNode, ProprioceptionNode
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation.utils import accumulate_time
from wild_visual_navigation_msgs.msg import RobotState
from wild_visual_navigation_msgs.srv import SaveLoadData, SaveLoadDataResponse
from wild_visual_navigation.utils import Timer
from wild_visual_navigation.utils import WVNMode
from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray
from std_srvs.srv import Trigger, TriggerResponse
from threading import Thread
from visualization_msgs.msg import Marker
import message_filters
import os
import rospy
import seaborn as sns
import tf
import torch
import numpy as np
from typing import Optional
import traceback

torch.cuda.empty_cache()


class WvnRosInterface:
    def __init__(self):
        # Timers to control the rate of the publishers
        self.last_image_ts = rospy.get_time()
        self.last_proprio_ts = rospy.get_time()

        # Read params
        self.read_params()

        # Visualization
        self.color_palette = sns.color_palette(self.colormap, as_cmap=True)

        # Initialize traversability estimator
        self.traversability_estimator = TraversabilityEstimator(
            device=self.device,
            segmentation_type=self.segmentation_type,
            feature_type=self.feature_type,
            max_distance=self.traversability_radius,
            image_distance_thr=self.image_graph_dist_thr,
            proprio_distance_thr=self.proprio_graph_dist_thr,
            optical_flow_estimator_type=self.optical_flow_estimator_type,
            mode=self.mode,
            running_store_folder=self.running_store_folder,
            exp_file=self.exp_file,
        )

        # Initialize traversability generator to process velocity commands
        self.supervision_generator = SupervisionGenerator(
            self.device,
            kf_process_cov=0.1,
            kf_meas_cov=10,
            kf_outlier_rejection="huber",
            kf_outlier_rejection_delta=0.5,
            sigmoid_slope=20,
            sigmoid_cutoff=0.25,  # 0.2
            untraversable_thr=self.untraversable_thr,  # 0.1
            time_horizon=0.05,
        )

        # Setup ros
        self.setup_ros(setup_fully=self.mode != WVNMode.EXTRACT_LABELS)

        # Launch processes
        print("─" * 80)

        # Setup slow threads
        print("Launching [learning] thread")
        if self.run_online_learning:
            self.learning_thread = Thread(target=self.learning_thread_loop, name="learning")
            self.learning_thread.start()
        print("[WVN] System ready")

    def __str__(self):
        s = "WvnRosInterface:"
        if hasattr(self, "time_summary"):
            for (k, v) in self.time_summary.items():
                n = self.n_summary[k]
                s += f"\n  {k}:".ljust(25) + f" {round(v,2)}ms  counts: {n} "
        return s

    def __del__(self):
        """Destructor
        Joins all the running threads
        """
        # Join threads
        if self.run_online_learning:
            self.learning_thread.join()

    @accumulate_time
    def read_params(self):
        """Reads all the parameters from the parameter server"""
        # Topics
        self.robot_state_topic = rospy.get_param("~robot_state_topic", "/wild_visual_navigation_node/robot_state")
        self.image_topic = rospy.get_param("~image_topic", "/alphasense_driver_ros/cam4/debayered")
        self.info_topic = rospy.get_param("~camera_info_topic", "/alphasense_driver_ros/cam4/camera_info")
        self.desired_twist_topic = rospy.get_param("~desired_twist_topic", "/log/state/desiredRobotTwist")

        # Frames
        self.fixed_frame = rospy.get_param("~fixed_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.camera_frame = rospy.get_param("~camera_frame", "cam4_sensor_frame_helper")
        self.footprint_frame = rospy.get_param("~footprint_frame", "footprint")

        # Robot size and specs
        self.robot_length = rospy.get_param("~robot_length", 1.0)
        self.robot_width = rospy.get_param("~robot_width", 0.6)
        self.robot_height = rospy.get_param("~robot_height", 0.3)
        self.robot_max_velocity = rospy.get_param("~robot_max_velocity", 0.8)

        # Traversability estimation params
        self.traversability_radius = rospy.get_param("~traversability_radius", 3.0)
        self.image_graph_dist_thr = rospy.get_param("~image_graph_dist_thr", 0.2)
        self.proprio_graph_dist_thr = rospy.get_param("~proprio_graph_dist_thr", 0.1)
        self.network_input_image_height = rospy.get_param("~network_input_image_height", 448)
        self.network_input_image_width = rospy.get_param("~network_input_image_width", 448)
        self.segmentation_type = rospy.get_param("~segmentation_type", "slic")
        self.feature_type = rospy.get_param("~feature_type", "dino")

        # Supervision Generator
        self.untraversable_thr = rospy.get_param("~untraversable_thr", 0)

        # Optical flow params
        self.optical_flow_estimator_type = rospy.get_param("~optical_flow_estimator_type", "sparse")

        # Threads
        self.run_online_learning = rospy.get_param("~run_online_learning", True)
        self.image_callback_rate = rospy.get_param("~image_callback_rate", 3)  # hertz
        self.proprio_callback_rate = rospy.get_param("~proprio_callback_rate", 4)  # hertz
        self.learning_thread_rate = rospy.get_param("~learning_thread_rate", 10)  # hertz

        # Data storage
        out_path = os.path.join(WVN_ROOT_DIR, "results")
        self.output_path = rospy.get_param("~output_path", out_path)
        self.mission_name = rospy.get_param("~mission_name", "default_mission")

        # Print timings
        self.print_image_callback_time = rospy.get_param("~print_image_callback_time", False)

        # Select mode: # debug, online, extract_labels
        self.use_debug_for_desired = rospy.get_param("~use_debug_for_desired", True)
        self.mode = WVNMode.from_string(rospy.get_param("~mode", "default"))
        self.running_store_folder = rospy.get_param("~running_store_folder", "nan")

        # Parse operation modes
        if self.mode == WVNMode.ONLINE:
            print("\nWARNING: online_mode enabled. The graph will not store any debug/training data such as images\n")

        elif self.mode == WVNMode.EXTRACT_LABELS:
            self.run_online_learning = False
            self.image_callback_rate = 3
            self.proprio_callback_rate = 4
            self.optical_flow_estimator_type = False
            self.image_graph_dist_thr = 0.2
            self.proprio_graph_dist_thr = 0.1

            os.makedirs(os.path.join(self.running_store_folder, "image"), exist_ok=True)
            os.makedirs(os.path.join(self.running_store_folder, "supervision_mask"), exist_ok=True)

        # Experiment file
        self.exp_file = rospy.get_param("~exp", "nan")

        # Torch device
        self.device = rospy.get_param("~device", "cuda")

        # Visualization
        self.colormap = rospy.get_param("~colormap", "RdYlBu")

    def setup_rosbag_replay(self, tf_listener):
        self.tf_listener = tf_listener

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        if setup_fully:
            # Initialize TF listener
            self.tf_listener = tf.TransformListener()

            # Robot state callback
            robot_state_sub = message_filters.Subscriber(self.robot_state_topic, RobotState)
            cache1 = message_filters.Cache(robot_state_sub, 1)
            desired_twist_sub = message_filters.Subscriber(self.desired_twist_topic, TwistStamped)
            cache2 = message_filters.Cache(desired_twist_sub, 1)

            self.robot_state_sub = message_filters.ApproximateTimeSynchronizer(
                [robot_state_sub, desired_twist_sub], queue_size=10, slop=0.1
            )
            self.robot_state_sub.registerCallback(self.robot_state_callback)

            # policy_debug_info_sub = message_filters.Subscriber("/debug_info", Float32MultiArray, queue_size=10)
            # self.robot_state_policy_debug_info_sub = message_filters.ApproximateTimeSynchronizer(
            #    [robot_state_sub, policy_debug_info_sub], queue_size=1, slop=9999999999999, allow_headerless=True,
            # )
            # self.robot_state_policy_debug_info_sub.registerCallback(self.robot_state_policy_debug_info_callback)

            # Image callback
            self.image_sub = message_filters.Subscriber(self.image_topic, Image)
            self.info_sub = message_filters.Subscriber(self.info_topic, CameraInfo)
            self.image_synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.info_sub], queue_size=2, slop=0.1
            )
            self.image_synchronizer.registerCallback(self.image_callback)

        # Publishers
        self.pub_image_labeled = rospy.Publisher(
            "/wild_visual_navigation_node/last_node_image_labeled", Image, queue_size=10
        )
        self.pub_image_mask = rospy.Publisher("/wild_visual_navigation_node/last_node_image_mask", Image, queue_size=10)
        self.pub_image_prediction_input = rospy.Publisher(
            "/wild_visual_navigation_node/current_prediction_input", Image, queue_size=10
        )
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
        self.pub_instant_traversability = rospy.Publisher(
            "/wild_visual_navigation_node/instant_traversability", Float32, queue_size=10
        )
        self.pub_training_loss = rospy.Publisher("/wild_visual_navigation_node/training_loss", Float32, queue_size=10)

        self.pub_traversability = rospy.Publisher(
            "/wild_visual_navigation_node/traversability_raw", Image, queue_size=10
        )
        self.pub_confidence = rospy.Publisher("/wild_visual_navigation_node/confidence_raw", Image, queue_size=10)
        self.pub_label = rospy.Publisher("/wild_visual_navigation_node/label_raw", Image, queue_size=10)
        self.pub_camera_info = rospy.Publisher("/wild_visual_navigation_node/camera_info", CameraInfo, queue_size=10)

        # Services
        # Like, reset graph or the like
        self.save_graph_service = rospy.Service("~save_graph", SaveLoadData, self.save_graph_callback)
        self.save_pickle_service = rospy.Service("~save_pickle", SaveLoadData, self.save_pickle_callback)
        self.save_checkpt_service = rospy.Service("~save_checkpoint", SaveLoadData, self.save_checkpoint_callback)
        self.load_checkpt_service = rospy.Service("~load_checkpoint", SaveLoadData, self.load_checkpoint_callback)

    @accumulate_time
    def save_graph_callback(self, req):
        """Service call to store the mission graph as a dataset

        Args:
            req (SaveLoadDataRequest): SaveLoadData obejct with the request

        Returns:
            res (SaveLoadDataResponse): Status of the request
        """
        if req.path == "" or req.mission_name == "":
            return SaveLoadDataResponse(
                success=False,
                message=f"Either output_path [{path}] or mission_name [{mission_name}] is empty. Please check and try again",
            )

        mission_path = os.path.join(req.path, req.mission_name)
        t = Thread(target=self.traversability_estimator.save_graph, args=(mission_path,))
        t.start()
        t.join()
        return SaveLoadDataResponse(success=True, message=f"Graph saved in {mission_path}")

    @accumulate_time
    def save_pickle_callback(self, req):
        """Service call to store the traversability estimator instance as a pickle file

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.path == "" or req.mission_name == "":
            return SaveLoadDataResponse(
                success=False,
                message=f"Either path [{path}] or mission_name [{mission_name}] is empty. Please check and try again",
            )

        mission_path = os.path.join(req.path, req.mission_name)
        pickle_file = "traversability_estimator.pickle"
        self.traversability_estimator.save(mission_path, pickle_file)
        return SaveLoadDataResponse(success=True, message=f"Pickle [{pickle_file}] saved in {mission_path}")

    @accumulate_time
    def save_checkpoint_callback(self, req):
        """Service call to store the learned model

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.path == "" or req.mission_name == "":
            return SaveLoadDataResponse(
                success=False,
                message=f"Either path [{path}] or mission_name [{mission_name}] is empty. Please check and try again",
            )

        mission_path = os.path.join(req.path, req.mission_name)
        checkpoint_file = "model_checkpoint.pt"
        self.traversability_estimator.save_checkpoint(mission_path, checkpoint_file)
        return SaveLoadDataResponse(success=True, message=f"Checkpoint [{checkpoint_file}] saved in {mission_path}")

    def load_checkpoint_callback(self, req):
        """Service call to load a learned model

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.path == "" or req.mission_name == "":
            return SaveLoadDataResponse(
                success=False,
                message=f"Either path [{path}] or mission_name [{mission_name}] is empty. Please check and try again",
            )

        mission_path = os.path.join(req.path, req.mission_name)
        checkpoint_file = "model_checkpoint.pt"
        self.traversability_estimator.load_checkpoint(mission_path, checkpoint_file)
        return SaveLoadDataResponse(success=True, message=f"Checkpoint [{checkpoint_file}] loaded successfully")

    @accumulate_time
    def query_tf(self, parent_frame: str, child_frame: str, stamp: Optional[rospy.Time] = None):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
        """

        if stamp is None:
            stamp = rospy.Time(0)
        # Wait for required tfs
        try:
            self.tf_listener.waitForTransform(parent_frame, child_frame, stamp, rospy.Duration(1.0))
        except Exception as e:
            print("Error in querry tf: ", e)
            return (None, None)

        try:
            (trans, rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, stamp)
            return (trans, rot)
        except Exception as e:
            print("Error in querry tf: ", e)
            # (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): avoid all errors
            rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)

    # def robot_state_policy_debug_info_callback(self, state_msg, debug_info_msg):
    #     desired_twist_msg = TwistStamped()
    #     desired_twist_msg.twist.linear.x = debug_info_msg.data[0]
    #     desired_twist_msg.twist.linear.y = debug_info_msg.data[1]
    #     desired_twist_msg.twist.angular.z = debug_info_msg.data[2]
    #     self.robot_state_callback(state_msg, desired_twist_msg)

    @accumulate_time
    def robot_state_callback(self, state_msg, desired_twist_msg: TwistStamped):
        """Main callback to process proprioceptive info (robot state)

        Args:
            state_msg (wild_visual_navigation_msgs/RobotState): Robot state message
            desired_twist_msg (geometry_msgs/TwistStamped): Desired twist message
        """
        try:
            ts = state_msg.header.stamp.to_sec()
            if abs(ts - self.last_proprio_ts) < 1.0 / self.proprio_callback_rate:
                return
            self.last_propio_ts = ts

            # Query transforms from TF
            suc, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(self.fixed_frame, self.base_frame, state_msg.header.stamp), device=self.device
            )
            if not suc:
                return

            suc, pose_footprint_in_base = rc.ros_tf_to_torch(
                self.query_tf(self.base_frame, self.footprint_frame, state_msg.header.stamp), device=self.device
            )
            if not suc:
                return

            # The footprint requires a correction: we use the same orientation as the base
            pose_footprint_in_base[:3, :3] = torch.eye(3, device=self.device)

            # Convert state to tensor
            proprio_tensor, proprio_labels = rc.wvn_robot_state_to_torch(state_msg, device=self.device)
            current_twist_tensor = rc.twist_stamped_to_torch(state_msg.twist, device=self.device)
            desired_twist_tensor = rc.twist_stamped_to_torch(desired_twist_msg, device=self.device)

            # Update traversability
            traversability, traversability_var, is_untraversable = self.supervision_generator.update_velocity_tracking(
                current_twist_tensor, desired_twist_tensor, velocities=["vx", "vy"]
            )

            # Create proprioceptive node for the graph
            proprio_node = ProprioceptionNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                pose_footprint_in_base=pose_footprint_in_base,
                twist_in_base=current_twist_tensor,
                width=self.robot_width,
                length=self.robot_length,
                height=self.robot_width,
                proprioception=proprio_tensor,
                traversability=traversability,
                traversability_var=traversability_var,
                is_untraversable=is_untraversable,
            )

            # Add node to the graph
            self.traversability_estimator.add_proprio_node(proprio_node)

            if self.mode != WVNMode.EXTRACT_LABELS:
                # Visualizations (45ms)
                self.visualize_proprioception()
        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)

    @accumulate_time
    def image_callback(self, image_msg: Image, info_msg: CameraInfo):
        """Main callback to process incoming images

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
        """

        print("Image callback")
        try:

            # Run the callback so as to match the desired rate
            ts = image_msg.header.stamp.to_sec()
            if abs(ts - self.last_image_ts) < 1.0 / self.image_callback_rate:
                return
            self.last_image_ts = ts

            # Query transforms from TF
            suc, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(self.fixed_frame, self.base_frame, image_msg.header.stamp), device=self.device
            )
            if not suc:
                return
            suc, pose_cam_in_base = rc.ros_tf_to_torch(
                self.query_tf(self.base_frame, self.camera_frame, image_msg.header.stamp), device=self.device
            )
            if not suc:
                return

            # Prepare image projector
            K, H, W = rc.ros_cam_info_to_tensors(info_msg, device=self.device)
            image_projector = ImageProjector(
                K=K, h=H, w=W, new_h=self.network_input_image_height, new_w=self.network_input_image_width
            )

            # Add image to base node
            # convert image message to torch image
            torch_image = rc.ros_image_to_torch(image_msg, device=self.device)
            torch_image = image_projector.resize_image(torch_image)

            # Create mission node for the graph
            mission_node = MissionNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                pose_cam_in_base=pose_cam_in_base,
                image=torch_image,
                image_projector=image_projector,
                correspondence=torch.zeros((1,)) if self.optical_flow_estimator_type != "sparse" else None,
            )

            with Timer("image_callback - add_mission_node"):
                # Add node to graph
                added_new_node = self.traversability_estimator.add_mission_node(mission_node)

            # Update prediction
            self.traversability_estimator.update_prediction(mission_node)

            with Timer("image_callback - update visualizations"):
                if self.mode != WVNMode.EXTRACT_LABELS:
                    # Publish current predictions
                    self.publish_predictions(mission_node, image_msg, info_msg, image_projector.scaled_camera_matrix)

                    # Publish supervision data depending on the mode
                    if self.mode != WVNMode.ONLINE:
                        self.visualize_mission()
                    else:
                        self.visualize_mission(fast=True)

            # If a new node was added, update the node is used to visualize the supervision signals
            if added_new_node:
                self.traversability_estimator.update_visualization_node()

            # Print callback time if required
            if self.print_image_callback_time:
                print(self)

        except Exception as e:
            traceback.print_exc()
            print("error image callback", e)

    @accumulate_time
    def publish_predictions(
        self, mission_node: MissionNode, image_msg: Image, info_msg: CameraInfo, scaled_camera_matrix: torch.Tensor
    ):
        """Publish predictions for the current image

        Args:
            mission_node (MissionNode): Mission node
        """
        # Publish predictions
        if mission_node is not None and self.run_online_learning:
            # Traversability
            out_trav = torch.zeros(mission_node.feature_segments.shape, device=self.device)
            out_conf = torch.zeros(mission_node.feature_segments.shape, device=self.device)

            for i in range(mission_node.prediction.shape[0]):
                out_trav[i == mission_node.feature_segments] = mission_node.prediction[i, 0]
                out_conf[i == mission_node.feature_segments] = mission_node.confidence[i]

            msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
            msg.header = image_msg.header
            msg.width = out_trav.shape[0]
            msg.height = out_trav.shape[1]
            self.pub_traversability.publish(msg)
            # out_conf[:,:] = 1
            # out_conf[int(out_conf.shape[0]/2):,:] = 0
            # out_conf[:,int(out_conf.shape[1]/2):] = 0
            msg = rc.numpy_to_ros_image(out_conf.cpu().numpy(), "passthrough")
            msg.header = image_msg.header
            msg.width = out_trav.shape[0]
            msg.height = out_trav.shape[1]
            self.pub_confidence.publish(msg)

            info_msg.width = out_trav.shape[0]
            info_msg.height = out_trav.shape[1]
            info_msg.K = scaled_camera_matrix[0, :3, :3].cpu().numpy().flatten().tolist()
            info_msg.P = scaled_camera_matrix[0, :3, :4].cpu().numpy().flatten().tolist()
            self.pub_camera_info.publish(info_msg)
            # self.pub_label.publish(rc.numpy_to_ros_image(np_labeled_image))

    @accumulate_time
    def learning_thread_loop(self):
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        # Set rate
        rate = rospy.Rate(self.learning_thread_rate)

        # Main loop
        while not rospy.is_shutdown():
            # Optimize model
            loss = self.traversability_estimator.train()

            # Publish loss
            self.pub_training_loss.publish(loss)
            rate.sleep()

    @accumulate_time
    def visualize_proprioception(self):
        """Publishes all the visualizations related to proprioceptive info,
        like footprints and the sliding graph
        """
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
        footprints_marker.color.a = 1.0
        footprints_marker.pose.orientation.w = 1.0
        footprints_marker.pose.position.x = 0.0
        footprints_marker.pose.position.y = 0.0
        footprints_marker.pose.position.z = 0.0

        last_points = [None, None]
        for node in self.traversability_estimator.get_proprio_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_base_in_world)
            proprio_graph_msg.poses.append(pose)

            # Color for traversability
            r, g, b, _ = self.color_palette(node.traversability.item())
            c = ColorRGBA(r, g, b, 0.95)

            # Rainbow path
            side_points = node.get_side_points()

            # if the last points are empty, fill and continue
            if None in last_points:
                for i in range(2):
                    last_points[i] = Point(
                        x=side_points[i, 0].item(), y=side_points[i, 1].item(), z=side_points[i, 2].item()
                    )
                continue
            else:
                # here we want to add 2 triangles: from the last saved points (lp)
                # and the current side points (sp):
                # triangle 1: [lp0, lp1, sp0]
                # triangle 2: [lp1, sp0, sp1]

                points_to_add = []
                # add the last points points
                for lp in last_points:
                    points_to_add.append(lp)
                # Add first from new side points
                points_to_add.append(
                    Point(x=side_points[i, 0].item(), y=side_points[i, 1].item(), z=side_points[i, 2].item())
                )
                # Add last of last points
                points_to_add.append(last_points[0])
                # Add new side points and update last points
                for i in range(2):
                    last_points[i] = Point(
                        x=side_points[i, 0].item(), y=side_points[i, 1].item(), z=side_points[i, 2].item()
                    )
                    points_to_add.append(last_points[i])

                # Add all the points and colors
                for p in points_to_add:
                    footprints_marker.points.append(p)
                    footprints_marker.colors.append(c)

            # Untraversable plane
            if node.is_untraversable:
                untraversable_plane = node.get_untraversable_plane(grid_size=2)
                N, D = untraversable_plane.shape
                for n in [0, 1, 3, 2, 0, 3]:  # this is a hack to show the triangles correctly
                    p = Point()
                    p.x = untraversable_plane[n, 0]
                    p.y = untraversable_plane[n, 1]
                    p.z = untraversable_plane[n, 2]
                    footprints_marker.points.append(p)
                    footprints_marker.colors.append(c)

        # Publish
        if len(footprints_marker.points) % 3 != 0:
            print(f"number of points for footprint is {len(footprints_marker.points)}")
            return
        self.pub_graph_footprints.publish(footprints_marker)
        self.pub_debug_proprio_graph.publish(proprio_graph_msg)

        # Publish latest traversability
        self.pub_instant_traversability.publish(self.supervision_generator.traversability)

    @accumulate_time
    def visualize_mission(self, fast: bool = False):
        """Publishes all the visualizations related to mission graph, like the graph
        itself, visual features, supervision signals, and traversability estimates
        """
        # Get current time for later
        now = rospy.Time.now()

        # Get visualization node
        vis_node = self.traversability_estimator.get_mission_node_for_visualization()

        if not fast:
            # Publish predictions
            if vis_node is not None and self.run_online_learning:
                with Timer("plot_mission_node_prediction"):
                    (
                        np_prediction_image,
                        np_uncertainty_image,
                    ) = self.traversability_estimator.plot_mission_node_prediction(vis_node)
                self.pub_image_prediction_input.publish(rc.torch_to_ros_image(vis_node.image))
                self.pub_image_prediction.publish(rc.numpy_to_ros_image(np_prediction_image))
                self.pub_image_prediction_uncertainty.publish(rc.numpy_to_ros_image(np_uncertainty_image))

            # Publish reprojections of last node in graph
            if vis_node is not None:
                with Timer("plot_mission_node_training"):
                    np_labeled_image, np_mask_image = self.traversability_estimator.plot_mission_node_training(vis_node)
                if np_labeled_image is None or np_mask_image is None:
                    return
                self.pub_image_labeled.publish(rc.numpy_to_ros_image(np_labeled_image))
                self.pub_image_mask.publish(rc.numpy_to_ros_image(np_mask_image))

        # Publish mission graph
        mission_graph_msg = Path()
        mission_graph_msg.header.frame_id = self.fixed_frame
        mission_graph_msg.header.stamp = now

        for node in self.traversability_estimator.get_mission_nodes():
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_cam_in_world)
            mission_graph_msg.poses.append(pose)

        self.pub_mission_graph.publish(mission_graph_msg)


if __name__ == "__main__":
    rospy.init_node("wild_visual_navigation_node")
    wvn = WvnRosInterface()
    rospy.spin()
