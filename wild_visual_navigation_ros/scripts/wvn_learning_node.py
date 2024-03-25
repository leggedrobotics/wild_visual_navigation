#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.supervision_generator import SupervisionGenerator
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import MissionNode, SupervisionNode
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation_ros.reload_rosparams import reload_rosparams
from wild_visual_navigation_msgs.msg import RobotState, SystemState, ImageFeatures
from wild_visual_navigation.visu import LearningVisualizer
from wild_visual_navigation_msgs.srv import (
    LoadCheckpoint,
    SaveCheckpoint,
    LoadCheckpointResponse,
    SaveCheckpointResponse,
)
from wild_visual_navigation.utils import WVNMode, create_experiment_folder
from wild_visual_navigation.cfg import ExperimentParams, RosLearningNodeParams

from std_srvs.srv import SetBool, Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker
import tf2_ros
import rospy
import message_filters

from pytictac import ClassTimer, ClassContextTimer, accumulate_time
from omegaconf import OmegaConf, read_write
from threading import Thread, Event
import os
import seaborn as sns
import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys


def time_func():
    return rospy.get_time()


class WvnLearning:
    def __init__(self, node_name):
        # Timers to control the rate of the publishers
        self._last_image_ts = time_func()
        self._last_supervision_ts = time_func()
        self._last_checkpoint_ts = time_func()
        self._setup_ready = False

        # Prepare variables
        self._node_name = node_name

        # Read params
        self.read_params()

        # Initialize camera handler for subscription/publishing
        self._system_events = {}

        # Setup ros
        self.setup_ros(setup_fully=self._ros_params.mode != WVNMode.EXTRACT_LABELS)

        # Visualization
        self._color_palette = sns.color_palette(self._ros_params.colormap, as_cmap=True)

        # Setup Mission Folder
        model_path = create_experiment_folder(self._params)

        with read_write(self._params):
            self._params.general.model_path = model_path

        # Initialize traversability estimator
        self._traversability_estimator = TraversabilityEstimator(
            params=self._params,
            device=self._ros_params.device,
            max_distance=self._ros_params.traversability_radius,
            image_distance_thr=self._ros_params.image_graph_dist_thr,
            supervision_distance_thr=self._ros_params.supervision_graph_dist_thr,
            min_samples_for_training=self._ros_params.min_samples_for_training,
            vis_node_index=self._ros_params.vis_node_index,
            mode=self._ros_params.mode,
            extraction_store_folder=self._ros_params.extraction_store_folder,
            anomaly_detection=self.anomaly_detection,
        )

        # Initialize traversability generator to process velocity commands
        self._supervision_generator = SupervisionGenerator(
            device=self._ros_params.device,
            kf_process_cov=0.1,
            kf_meas_cov=10,
            kf_outlier_rejection="huber",
            kf_outlier_rejection_delta=0.5,
            sigmoid_slope=20,
            sigmoid_cutoff=0.25,  # 0.2
            untraversable_thr=self._ros_params.untraversable_thr,  # 0.1
            time_horizon=0.05,
            graph_max_length=1,
        )

        # Setup Timer if needed
        self._timer = ClassTimer(
            objects=[
                self,
                self._traversability_estimator,
                self._traversability_estimator._visualizer,
                self._supervision_generator,
            ],
            names=[
                "WVN",
                "TraversabilityEstimator",
                "Visualizer",
                "SupervisionGenerator",
            ],
            enabled=(
                self._ros_params.print_image_callback_time
                or self._ros_params.print_supervision_callback_time
                or self._ros_params.log_time
            ),
        )

        # Register shutdown callbacks
        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

        # Launch processes
        print("-" * 80)
        self._setup_ready = True
        rospy.loginfo(f"[{self._node_name}] Launching [learning] thread")
        if self._ros_params.mode != WVNMode.EXTRACT_LABELS:
            self._learning_thread_stop_event = Event()
            self.learning_thread = Thread(target=self.learning_thread_loop, name="learning")
            self.learning_thread.start()

        # self.logging_thread_stop_event = Event()
        # self.logging_thread = Thread(target=self.logging_thread_loop, name="logging")
        # self.logging_thread.start()
        rospy.loginfo(f"[{self._node_name}] [WVN] System ready")

    def shutdown_callback(self, *args, **kwargs):
        # Write stuff to files
        rospy.logwarn("Shutdown callback called")
        if self._ros_params.mode != WVNMode.EXTRACT_LABELS:
            self._learning_thread_stop_event.set()
            # self.logging_thread_stop_event.set()

        print(f"[{self._node_name}] Storing learned checkpoint...", end="")
        self._traversability_estimator.save_checkpoint(self._params.general.model_path, "last_checkpoint.pt")
        print("done")

        if self._ros_params.log_time:
            print(f"[{self._node_name}] Storing timer data...", end="")
            self._timer.store(folder=self._params.general.model_path)
            print("done")

        print(f"[{self._node_name}] Joining learning thread...", end="")
        if self._ros_params.mode != WVNMode.EXTRACT_LABELS:
            self._learning_thread_stop_event.set()
            self.learning_thread.join()

            # self.logging_thread_stop_event.set()
            # self.logging_thread.join()
        print("done")

        rospy.signal_shutdown(f"[{self._node_name}] Wild Visual Navigation killed {args}")
        sys.exit(0)

    @accumulate_time
    def read_params(self):
        """Reads all the parameters from the parameter server"""
        self._params = OmegaConf.structured(ExperimentParams)
        self._ros_params = OmegaConf.structured(RosLearningNodeParams)

        # Override the empty dataclass with values from ros parmeter server
        with read_write(self._ros_params):
            for k in self._ros_params.keys():
                self._ros_params[k] = rospy.get_param(f"~{k}")

        self._ros_params.robot_height = rospy.get_param("~robot_height")  # TODO robot_height currently not used

        with read_write(self._ros_params):
            self._ros_params.mode = WVNMode.from_string(self._ros_params.mode)

        with read_write(self._params):
            self._params.general.name = self._ros_params.mission_name
            self._params.general.timestamp = self._ros_params.mission_timestamp
            self._params.general.log_confidence = self._ros_params.log_confidence
            self._params.loss.confidence_std_factor = self._ros_params.confidence_std_factor
            self._params.loss.w_temp = 0

        # Parse operation modes
        if self._ros_params.mode == WVNMode.ONLINE:
            rospy.logwarn(
                f"[{self._node_name}] WARNING: online_mode enabled. The graph will not store any debug/training data such as images\n"
            )

        elif self._ros_params.mode == WVNMode.EXTRACT_LABELS:
            with read_write(self._ros_params):
                # TODO verify if this is needed
                self._ros_params.image_callback_rate = 3
                self._ros_params.supervision_callback_rate = 4
                self._ros_params.image_graph_dist_thr = 0.2
                self._ros_params.supervision_graph_dist_thr = 0.1
            os.makedirs(
                os.path.join(self._ros_params.extraction_store_folder, "image"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(self._ros_params.extraction_store_folder, "supervision_mask"),
                exist_ok=True,
            )

        self._step = -1
        self._step_time = time_func()
        self.anomaly_detection = self._params.model.name == "LinearRnvp"

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        if setup_fully:
            # Initialize TF listener
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            # Robot state callback
            robot_state_sub = message_filters.Subscriber(self._ros_params.robot_state_topic, RobotState)
            cache1 = message_filters.Cache(robot_state_sub, 10)  # noqa: F841
            desired_twist_sub = message_filters.Subscriber(self._ros_params.desired_twist_topic, TwistStamped)
            cache2 = message_filters.Cache(desired_twist_sub, 10)  # noqa: F841

            self._robot_state_sub = message_filters.ApproximateTimeSynchronizer(
                [robot_state_sub, desired_twist_sub], queue_size=10, slop=0.5
            )

            rospy.loginfo(
                f"[{self._node_name}] Start waiting for RobotState topic {self._ros_params.robot_state_topic} being published!"
            )
            rospy.wait_for_message(self._ros_params.robot_state_topic, RobotState)
            rospy.loginfo(
                f"[{self._node_name}] Start waiting for TwistStamped topic {self._ros_params.desired_twist_topic} being published!"
            )
            rospy.wait_for_message(self._ros_params.desired_twist_topic, TwistStamped)
            self._robot_state_sub.registerCallback(self.robot_state_callback)

            self._camera_handler = {}
            # Image callback
            for cam in self._ros_params.camera_topics:
                # Initialize camera handler for given cam
                self._camera_handler[cam] = {}
                # Store camera name
                self._ros_params.camera_topics[cam]["name"] = cam

                # Set subscribers
                if self._ros_params.mode == WVNMode.DEBUG:
                    # In debug mode additionally send the image to the callback function
                    self._visualizer = LearningVisualizer()

                    imagefeat_sub = message_filters.Subscriber(
                        f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures
                    )
                    info_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo)
                    image_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/image_input", Image)
                    sync = message_filters.ApproximateTimeSynchronizer(
                        [imagefeat_sub, info_sub, image_sub], queue_size=4, slop=0.5
                    )
                    sync.registerCallback(self.imagefeat_callback, self._ros_params.camera_topics[cam])

                    last_image_overlay_pub = rospy.Publisher(
                        f"/wild_visual_navigation_node/{cam}/debug/last_node_image_overlay",
                        Image,
                        queue_size=10,
                    )

                    self._camera_handler[cam]["debug"] = {}
                    self._camera_handler[cam]["debug"]["image_overlay"] = last_image_overlay_pub

                else:
                    print(f"/wild_visual_navigation_node/{cam}/feat")
                    imagefeat_sub = message_filters.Subscriber(
                        f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures
                    )
                    info_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo)
                    sync = message_filters.ApproximateTimeSynchronizer(
                        [imagefeat_sub, info_sub], queue_size=4, slop=0.5
                    )
                    sync.registerCallback(self.imagefeat_callback, self._ros_params.camera_topics[cam])

            # Wait for features message to determine the input size of the model
            cam = list(self._ros_params.camera_topics.keys())[0]

            exists_camera_used_for_training = False
            for cam in self._ros_params.camera_topics:
                rospy.loginfo(f"[{self._node_name}] Waiting for feat topic {cam}...")
                if self._ros_params.camera_topics[cam]["use_for_training"]:
                    feat_msg = rospy.wait_for_message(f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures)
                    exists_camera_used_for_training = True

            if not exists_camera_used_for_training:
                rospy.logerror("No camera selected for training")
                sys.exit(-1)

            feature_dim = int(feat_msg.features.layout.dim[1].size)
            # Modify the parameters
            with read_write(self._params):
                self._params.model.simple_mlp_cfg.input_size = feature_dim
                self._params.model.double_mlp_cfg.input_size = feature_dim
                self._params.model.simple_gcn_cfg.input_size = feature_dim
                self._params.model.linear_rnvp_cfg.input_size = feature_dim
            rospy.loginfo(f"[{self._node_name}] Done")

        # 3D outputs
        self._pub_debug_supervision_graph = rospy.Publisher(
            "/wild_visual_navigation_node/supervision_graph", Path, queue_size=10
        )
        self._pub_mission_graph = rospy.Publisher("/wild_visual_navigation_node/mission_graph", Path, queue_size=10)
        self._pub_graph_footprints = rospy.Publisher(
            "/wild_visual_navigation_node/graph_footprints", Marker, queue_size=10
        )
        # 1D outputs
        self._pub_instant_traversability = rospy.Publisher(
            "/wild_visual_navigation_node/instant_traversability",
            Float32,
            queue_size=10,
        )
        self._pub_system_state = rospy.Publisher(
            "/wild_visual_navigation_node/system_state", SystemState, queue_size=10
        )

        # Services
        # Like, reset graph or the like
        self._save_checkpt_service = rospy.Service("~save_checkpoint", SaveCheckpoint, self.save_checkpoint_callback)
        self._load_checkpt_service = rospy.Service("~load_checkpoint", LoadCheckpoint, self.load_checkpoint_callback)

        self._pause_learning_service = rospy.Service("~pause_learning", SetBool, self.pause_learning_callback)
        self._reset_service = rospy.Service("~reset", Trigger, self.reset_callback)

    @accumulate_time
    def learning_thread_loop(self):
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        # Set rate
        rate = rospy.Rate(self._ros_params.learning_thread_rate)
        # Learning loop
        while True:
            self._system_events["learning_thread_loop"] = {
                "time": time_func(),
                "value": "running",
            }
            self._learning_thread_stop_event.wait(timeout=0.01)
            if self._learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            # Optimize model
            with ClassContextTimer(parent_obj=self, block_name="training_step_time"):
                res = self._traversability_estimator.train()

            if self._step != self._traversability_estimator.step:
                self._step_time = time_func()
                self._step = self._traversability_estimator.step

            # Publish loss
            system_state = SystemState()
            for k in res.keys():
                if hasattr(system_state, k):
                    setattr(system_state, k, res[k])

            system_state.pause_learning = self._traversability_estimator.pause_learning
            system_state.mode = self._ros_params.mode.value
            system_state.step = self._step
            self._pub_system_state.publish(system_state)

            # Get current weights
            new_model_state_dict = self._traversability_estimator._model.state_dict()

            # Check the rate
            ts = time_func()
            if abs(ts - self._last_checkpoint_ts) > 1.0 / self._ros_params.load_save_checkpoint_rate:
                cg = self._traversability_estimator._traversability_loss._confidence_generator
                new_model_state_dict["confidence_generator"] = cg.get_dict()

                fn = os.path.join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
                if os.path.exists(fn):
                    os.remove(fn)
                torch.save(new_model_state_dict, fn)
                self._last_checkpoint_ts = ts
                print(
                    "Update model. Valid Nodes: ",
                    self._traversability_estimator._mission_graph.get_num_valid_nodes(),
                    " steps: ",
                    self._traversability_estimator._step,
                )

            rate.sleep()

        self._system_events["learning_thread_loop"] = {
            "time": time_func(),
            "value": "finished",
        }
        self._learning_thread_stop_event.clear()

    def logging_thread_loop(self):
        rate = rospy.Rate(self._ros_params.logging_thread_rate)

        # Learning loop
        while True:
            self._learning_thread_stop_event.wait(timeout=0.01)
            if self._learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped logging thread")
                break

            current_time = time_func()
            tmp = self._system_events.copy()
            rospy.loginfo("System Events:")
            for k, v in tmp.items():
                value = v["value"]
                msg = (
                    (k + ": ").ljust(35, " ")
                    + (str(round(current_time - v["time"], 4)) + "s ").ljust(10, " ")
                    + f" {value}"
                )
                rospy.loginfo(msg)
                rate.sleep()
            rospy.loginfo("--------------")
        self._learning_thread_stop_event.clear()

    @accumulate_time
    def robot_state_callback(self, state_msg, desired_twist_msg: TwistStamped):
        """Main callback to process supervision info (robot state)

        Args:
            state_msg (wild_visual_navigation_msgs/RobotState): Robot state message
            desired_twist_msg (geometry_msgs/TwistStamped): Desired twist message
        """
        if not self._setup_ready:
            return

        self._system_events["robot_state_callback_received"] = {
            "time": time_func(),
            "value": "message received",
        }
        try:
            ts = state_msg.header.stamp.to_sec()
            if abs(ts - self._last_supervision_ts) < 1.0 / self._ros_params.supervision_callback_rate:
                self._system_events["robot_state_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to rate",
                }
                return
            self._last_supervision_ts = ts

            # Query transforms from TF
            success, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.fixed_frame,
                    self._ros_params.base_frame,
                    state_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["robot_state_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_base_in_world",
                }
                return

            success, pose_footprint_in_base = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.base_frame,
                    self._ros_params.footprint_frame,
                    state_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["robot_state_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_footprint_in_base",
                }
                return

            # The footprint requires a correction: we use the same orientation as the base
            pose_footprint_in_base[:3, :3] = torch.eye(3, device=self._ros_params.device)

            # Convert state to tensor
            supervision_tensor, supervision_labels = rc.wvn_robot_state_to_torch(
                state_msg, device=self._ros_params.device
            )
            current_twist_tensor = rc.twist_stamped_to_torch(state_msg.twist, device=self._ros_params.device)
            desired_twist_tensor = rc.twist_stamped_to_torch(desired_twist_msg, device=self._ros_params.device)

            # Update traversability
            (
                traversability,
                traversability_var,
                is_untraversable,
            ) = self._supervision_generator.update_velocity_tracking(
                current_twist_tensor, desired_twist_tensor, velocities=["vx", "vy"]
            )

            # Create supervision node for the graph
            supervision_node = SupervisionNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                pose_footprint_in_base=pose_footprint_in_base,
                twist_in_base=current_twist_tensor,
                desired_twist_in_base=desired_twist_tensor,
                width=self._ros_params.robot_width,
                length=self._ros_params.robot_length,
                height=self._ros_params.robot_height,
                supervision=supervision_tensor,
                traversability=traversability,
                traversability_var=traversability_var,
                is_untraversable=is_untraversable,
            )

            # Add node to the graph
            self._traversability_estimator.add_supervision_node(supervision_node)

            if self._ros_params.mode == WVNMode.DEBUG or self._ros_params.mode == WVNMode.ONLINE:
                self.visualize_supervision()

            if self._ros_params.print_supervision_callback_time:
                print(f"[{self._node_name}]\n{self._timer}")

            self._system_events["robot_state_callback_state"] = {
                "time": time_func(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            rospy.logerr(f"[{self._node_name}] error state callback", e)
            self._system_events["robot_state_callback_state"] = {
                "time": time_func(),
                "value": f"failed to execute {e}",
            }

            raise Exception("Error in robot state callback")

    @accumulate_time
    def imagefeat_callback(self, *args):
        """Main callback to process incoming images

        Args:
            imagefeat_msg (wild_visual_navigation_msg/ImageFeatures): Incoming imagefeatures
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
        """
        if not self._setup_ready:
            return

        if self._ros_params.mode == WVNMode.DEBUG:
            assert len(args) == 4
            imagefeat_msg, info_msg, image_msg, camera_options = tuple(args)
        else:
            assert len(args) == 3
            imagefeat_msg, info_msg, camera_options = tuple(args)

        self._system_events["image_callback_received"] = {
            "time": time_func(),
            "value": "message received",
        }

        if self._ros_params.verbose:
            print(f"[{self._node_name}] Image callback: {camera_options['name']}... ", end="")

        try:
            # Run the callback so as to match the desired rate
            ts = imagefeat_msg.header.stamp.to_sec()
            if abs(ts - self._last_image_ts) < 1.0 / self._ros_params.image_callback_rate:
                return
            self._last_image_ts = ts

            # Query transforms from TF
            success, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.fixed_frame,
                    self._ros_params.base_frame,
                    imagefeat_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["image_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_base_in_world",
                }
                return

            success, pose_cam_in_base = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.base_frame,
                    imagefeat_msg.header.frame_id,
                    imagefeat_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["image_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_cam_in_base",
                }
                return

            # Prepare image projector
            K, H, W = rc.ros_cam_info_to_tensors(info_msg, device=self._ros_params.device)
            image_projector = ImageProjector(
                K=K,
                h=H,
                w=W,
                new_h=self._ros_params.network_input_image_height,
                new_w=self._ros_params.network_input_image_width,
            )
            # Add image to base node
            # convert image message to torch image
            feature_segments = rc.ros_image_to_torch(
                imagefeat_msg.feature_segments,
                desired_encoding="passthrough",
                device=self._ros_params.device,
            ).clone()
            h_small, w_small = feature_segments.shape[1:3]

            torch_image = None
            # convert image message to torch image
            if self._ros_params.mode == WVNMode.DEBUG:
                torch_image = rc.ros_image_to_torch(
                    image_msg,
                    desired_encoding="passthrough",
                    device=self._ros_params.device,
                ).clone()

            # Create mission node for the graph
            mission_node = MissionNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                pose_cam_in_base=pose_cam_in_base,
                image=torch_image,
                image_projector=image_projector,
                camera_name=camera_options["name"],
                use_for_training=camera_options["use_for_training"],
            )
            ma = imagefeat_msg.features
            dims = tuple(map(lambda x: x.size, ma.layout.dim))
            mission_node.features = torch.from_numpy(
                np.array(ma.data, dtype=float).reshape(dims).astype(np.float32)
            ).to(self._ros_params.device)
            mission_node.feature_segments = feature_segments[0]

            # Add node to graph
            added_new_node = self._traversability_estimator.add_mission_node(mission_node)

            if self._ros_params.mode == WVNMode.DEBUG:
                # Publish current predictions

                # Publish supervision data depending on the mode
                self.visualize_image_overlay()

                if added_new_node:
                    self._traversability_estimator.update_visualization_node()

                self.visualize_mission_graph()

            # Print callback time if required
            if self._ros_params.print_image_callback_time:
                rospy.loginfo(f"[{self._node_name}]\n{self._timer}")

            self._system_events["image_callback_state"] = {
                "time": time_func(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            rospy.logerr(f"[{self._node_name}] error image callback", e)
            self._system_events["image_callback_state"] = {
                "time": time_func(),
                "value": f"failed to execute {e}",
            }
            raise Exception("Error in image callback")

    @accumulate_time
    def visualize_supervision(self):
        """Publishes all the visualizations related to supervision info,
        like footprints and the sliding graph
        """
        # Get current time for later
        now = rospy.Time.now()

        supervision_graph_msg = Path()
        supervision_graph_msg.header.frame_id = self._ros_params.fixed_frame
        supervision_graph_msg.header.stamp = now

        # Footprints
        footprints_marker = Marker()
        footprints_marker.id = 0
        footprints_marker.ns = "footprints"
        footprints_marker.header.frame_id = self._ros_params.fixed_frame
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
        for node in self._traversability_estimator.get_supervision_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self._ros_params.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_base_in_world)
            supervision_graph_msg.poses.append(pose)

            # Color for traversability
            r, g, b, _ = self._color_palette(node.traversability.item())
            c = ColorRGBA(r, g, b, 0.95)

            # Rainbow path
            side_points = node.get_side_points()

            # if the last points are empty, fill and continue
            if None in last_points:
                for i in range(2):
                    last_points[i] = Point(
                        x=side_points[i, 0].item(),
                        y=side_points[i, 1].item(),
                        z=side_points[i, 2].item(),
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
                    Point(
                        x=side_points[i, 0].item(),
                        y=side_points[i, 1].item(),
                        z=side_points[i, 2].item(),
                    )
                )
                # Add last of last points
                points_to_add.append(last_points[0])
                # Add new side points and update last points
                for i in range(2):
                    last_points[i] = Point(
                        x=side_points[i, 0].item(),
                        y=side_points[i, 1].item(),
                        z=side_points[i, 2].item(),
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
                # the following is a 'hack' to show the triangles correctly
                for n in [0, 1, 3, 2, 0, 3]:
                    p = Point()
                    p.x = untraversable_plane[n, 0]
                    p.y = untraversable_plane[n, 1]
                    p.z = untraversable_plane[n, 2]
                    footprints_marker.points.append(p)
                    footprints_marker.colors.append(c)

        # Publish
        if len(footprints_marker.points) % 3 != 0:
            if self._ros_params.verbose:
                rospy.loginfo(f"[{self._node_name}] number of points for footprint is {len(footprints_marker.points)}")
            return
        self._pub_graph_footprints.publish(footprints_marker)
        self._pub_debug_supervision_graph.publish(supervision_graph_msg)

        # Publish latest traversability
        self._pub_instant_traversability.publish(self._supervision_generator.traversability)
        self._system_events["visualize_supervision"] = {
            "time": time_func(),
            "value": f"executed successfully",
        }

    @accumulate_time
    def visualize_mission_graph(self):
        """Publishes all the visualizations related to the mission graph"""
        # Get current time for later
        now = rospy.Time.now()

        # Publish mission graph
        mission_graph_msg = Path()
        mission_graph_msg.header.frame_id = self._ros_params.fixed_frame
        mission_graph_msg.header.stamp = now

        for node in self._traversability_estimator.get_mission_nodes():
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self._ros_params.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_cam_in_world)
            mission_graph_msg.poses.append(pose)

        self._pub_mission_graph.publish(mission_graph_msg)

    @accumulate_time
    def visualize_image_overlay(self):
        """Publishes all the debugging, slow visualizations"""

        # Get visualization node
        vis_node = self._traversability_estimator.get_mission_node_for_visualization()

        # Publish reprojections of last node in graph
        if vis_node is not None:
            cam = vis_node.camera_name
            torch_image = vis_node._image
            torch_mask = vis_node._supervision_mask
            torch_mask = torch.nan_to_num(torch_mask.nanmean(axis=0)) != 0
            torch_mask = torch_mask.float()

            image_out = self._visualizer.plot_detectron_classification(torch_image, torch_mask, cmap="Blues")
            self._camera_handler[cam]["debug"]["image_overlay"].publish(rc.numpy_to_ros_image(image_out))

    def pause_learning_callback(self, req):
        """Start and stop the network training"""
        prev_state = self._traversability_estimator.pause_learning
        self._traversability_estimator.pause_learning = req.data
        if not req.data and prev_state:
            message = "Resume training!"
        elif req.data and prev_state:
            message = "Training was already paused!"
        elif not req.data and not prev_state:
            message = "Training was already running!"
        elif req.data and not prev_state:
            message = "Pause training!"
        message += f" Updated the network for {self._traversability_estimator.step} steps"

        return True, message

    def reset_callback(self, req):
        """Resets the system"""
        rospy.logwarn(f"[{self._node_name}] System reset!")

        print(f"[{self._node_name}] Storing learned checkpoint...", end="")
        self._traversability_estimator.save_checkpoint(self._params.general.model_path, "last_checkpoint.pt")
        print("done")

        if self._ros_params.log_time:
            print(f"[{self._node_name}] Storing timer data...", end="")
            self._timer.store(folder=self._params.general.model_path)
            print("done")

        # Create new mission folder
        create_experiment_folder(self._params)

        # Reset traversability estimator
        self._traversability_estimator.reset()

        print(f"[{self._node_name}] Reset done")
        return TriggerResponse(True, "Reset done!")

    @accumulate_time
    def save_checkpoint_callback(self, req):
        """Service call to store the learned checkpoint

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.checkpoint_name == "":
            req.checkpoint_name = "last_checkpoint.pt"

        if req.mission_path == "":
            message = f"[WARNING] Store checkpoint {req.checkpoint_name} default mission path: {self._params.general.model_path}/{req.checkpoint_name}"
            req.mission_path = self._params.general.model_path
        else:
            message = f"Store checkpoint {req.checkpoint_name} to: {req.mission_path}/{req.checkpoint_name}"

        self._traversability_estimator.save_checkpoint(req.mission_path, req.checkpoint_name)
        return SaveCheckpointResponse(success=True, message=message)

    def load_checkpoint_callback(self, req):
        """Service call to load a learned checkpoint

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.checkpoint_path == "":
            return LoadCheckpointResponse(
                success=False,
                message=f"Path [{req.checkpoint_path}] is empty. Please check and try again",
            )
        checkpoint_path = req.checkpoint_path
        self._traversability_estimator.load_checkpoint(checkpoint_path)
        return LoadCheckpointResponse(success=True, message=f"Checkpoint [{checkpoint_path}] loaded successfully")

    @accumulate_time
    def query_tf(self, parent_frame: str, child_frame: str, stamp: Optional[rospy.Time] = None):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
        """

        if stamp is None:
            stamp = rospy.Time(0)

        try:
            res = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp, timeout=rospy.Duration(0.03))
            trans = (
                res.transform.translation.x,
                res.transform.translation.y,
                res.transform.translation.z,
            )
            rot = np.array(
                [
                    res.transform.rotation.x,
                    res.transform.rotation.y,
                    res.transform.rotation.z,
                    res.transform.rotation.w,
                ]
            )
            rot /= np.linalg.norm(rot)
            return (trans, tuple(rot))
        except Exception:
            if self._ros_params.verbose:
                # print("Error in query tf: ", e)
                rospy.logwarn(f"[{self._node_name}] Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)


if __name__ == "__main__":
    fn = os.path.join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
    if os.path.exists(fn):
        os.remove(fn)

    node_name = "wvn_learning_node"
    rospy.init_node(node_name)

    reload_rosparams(
        enabled=rospy.get_param("~reload_default_params", True),
        node_name=node_name,
        camera_cfg="wide_angle_dual",
    )
    wvn = WvnLearning(node_name)
    rospy.spin()
