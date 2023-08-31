from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.supervision_generator import SupervisionGenerator
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import MissionNode, ProprioceptionNode
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation_msgs.msg import RobotState, SystemState, ImageFeatures
from wild_visual_navigation_msgs.srv import (
    LoadCheckpoint,
    SaveCheckpoint,
    LoadCheckpointResponse,
    SaveCheckpointResponse,
)
from wild_visual_navigation.utils import WVNMode
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.utils import override_params
from wild_visual_navigation.learning.utils import load_yaml, load_env, create_experiment_folder


from std_srvs.srv import SetBool, Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker
import tf2_ros
import rospy
import message_filters

from pytictac import ClassTimer, ClassContextTimer, accumulate_time
from threading import Thread, Event
import os
import seaborn as sns
import torch
import dataclasses
import numpy as np
from typing import Optional
import traceback
import signal
import sys


if torch.cuda.is_available():
    torch.cuda.empty_cache()


class WvnLearning:
    def __init__(self):
        # Timers to control the rate of the publishers
        self.last_image_ts = rospy.get_time()
        self.last_proprio_ts = rospy.get_time()

        # Read params
        self.read_params()
        self.anomaly_detection = self.exp_cfg["model"]["name"] == "LinearRnvp"

        # Visualization
        self.color_palette = sns.color_palette(self.colormap, as_cmap=True)

        # Setup Mission Folder
        create_experiment_folder(self.params, load_env())

        # Initialize traversability estimator
        self.traversability_estimator = TraversabilityEstimator(
            params=self.params,
            device=self.device,
            image_size=self.network_input_image_height,  # Note: we assume height == width
            segmentation_type=self.segmentation_type,
            feature_type="none",
            max_distance=self.traversability_radius,
            image_distance_thr=self.image_graph_dist_thr,
            proprio_distance_thr=self.proprio_graph_dist_thr,
            min_samples_for_training=self.min_samples_for_training,
            vis_node_index=self.vis_node_index,
            mode=self.mode,
            extraction_store_folder=self.extraction_store_folder,
            patch_size=self.dino_patch_size,
            scale_traversability=self.scale_traversability,
            anomaly_detection=self.anomaly_detection,
            vis_training_samples=self.vis_training_samples,
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
        # Initialize camera handler for subscription/publishing
        self.system_events = {}
        # Setup ros
        self.setup_ros(setup_fully=self.mode != WVNMode.EXTRACT_LABELS)

        # Setup Timer if needed
        self.timer = ClassTimer(
            objects=[
                self,
                self.traversability_estimator,
                self.traversability_estimator._visualizer,
                self.supervision_generator,
            ],
            names=["WVN", "TraversabilityEstimator", "Visualizer", "SupervisionGenerator"],
            enabled=(self.print_image_callback_time or self.print_proprio_callback_time or self.log_time),
        )

        # Register shotdown callbacks
        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

        # Launch processes
        print("-" * 80)
        print("Launching [learning] thread")
        if self.mode != WVNMode.EXTRACT_LABELS:
            self.learning_thread_stop_event = Event()
            self.learning_thread = Thread(target=self.learning_thread_loop, name="learning")
            self.learning_thread.start()

            # self.logging_thread_stop_event = Event()
            # self.logging_thread = Thread(target=self.logging_thread_loop, name="logging")
            # self.logging_thread.start()
        print("[WVN] System ready")

    def shutdown_callback(self, *args, **kwargs):
        # Write stuff to files
        rospy.logwarn("Shutdown callback called")
        if self.mode != WVNMode.EXTRACT_LABELS:
            self.learning_thread_stop_event.set()
            self.logging_thread_stop_event.set()

        print("Storing learned checkpoint...", end="")
        self.traversability_estimator.save_checkpoint(self.params.general.model_path, "last_checkpoint.pt")
        print("done")

        if self.log_time:
            print("Storing timer data...", end="")
            self.timer.store(folder=self.params.general.model_path)
            print("done")

        print("Joining learning thread...", end="")
        if self.mode != WVNMode.EXTRACT_LABELS:
            self.learning_thread_stop_event.set()
            self.learning_thread.join()

            self.logging_thread_stop_event.set()
            self.logging_thread.join()
        print("done")

        rospy.signal_shutdown(f"Wild Visual Navigation killed {args}")
        sys.exit(0)

    @accumulate_time
    def learning_thread_loop(self):
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        # Set rate
        rate = rospy.Rate(self.learning_thread_rate)
        i = 0
        # Learning loop
        while True:
            self.system_events["learning_thread_loop"] = {"time": rospy.get_time(), "value": "running"}
            self.learning_thread_stop_event.wait(timeout=0.01)
            if self.learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            # Optimize model
            with ClassContextTimer(parent_obj=self, block_name="training_step_time"):
                res = self.traversability_estimator.train()

            if self.step != self.traversability_estimator.step:
                self.step_time = rospy.get_time()
                self.step = self.traversability_estimator.step

            # Publish loss
            system_state = SystemState()
            for k in res.keys():
                if hasattr(system_state, k):
                    setattr(system_state, k, res[k])

            system_state.pause_learning = self.traversability_estimator.pause_learning
            system_state.mode = self.mode.value
            system_state.step = self.step
            self.pub_system_state.publish(system_state)

            rate.sleep()
            if i % 10 == 0:
                res = self.traversability_estimator._model.state_dict()

                # Compute ROC Threshold
                if self.scale_traversability:
                    if self.traversability_estimator._auxiliary_training_roc._update_count != 0:
                        try:
                            fpr, tpr, thresholds = self.traversability_estimator._auxiliary_training_roc.compute()
                            index = torch.where(fpr > self.scale_traversability_max_fpr)[0][0]
                            traversability_threshold = thresholds[index]
                        except:
                            traversability_threshold = 0.5
                    else:
                        traversability_threshold = 0.5

                    res["traversability_threshold"] = traversability_threshold
                    cg = self.traversability_estimator._traversability_loss._confidence_generator
                    res["confidence_generator"] = cg.get_dict()

                os.system(f"rm {WVN_ROOT_DIR}/tmp_state_dict2.pt")
                torch.save(res, f"{WVN_ROOT_DIR}/tmp_state_dict2.pt")
            i += 1

        self.system_events["learning_thread_loop"] = {"time": rospy.get_time(), "value": "finished"}
        self.learning_thread_stop_event.clear()

    def logging_thread_loop(self):
        rate = rospy.Rate(self.logging_thread_rate)
        # Learning loop
        while True:
            self.learning_thread_stop_event.wait(timeout=0.01)
            if self.learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped logging thread")
                break

            current_time = rospy.get_time()
            tmp = self.system_events.copy()
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
        self.learning_thread_stop_event.clear()

    @accumulate_time
    def read_params(self):
        """Reads all the parameters from the parameter server"""
        # Topics
        self.robot_state_topic = rospy.get_param("~robot_state_topic")
        self.desired_twist_topic = rospy.get_param("~desired_twist_topic")
        self.camera_topics = rospy.get_param("~camera_topics")

        # Frames
        self.fixed_frame = rospy.get_param("~fixed_frame")
        self.base_frame = rospy.get_param("~base_frame")
        self.footprint_frame = rospy.get_param("~footprint_frame")

        # Robot size and specs
        self.robot_length = rospy.get_param("~robot_length")
        self.robot_width = rospy.get_param("~robot_width")
        self.robot_height = rospy.get_param("~robot_height")

        # Traversability estimation params
        self.traversability_radius = rospy.get_param("~traversability_radius")
        self.image_graph_dist_thr = rospy.get_param("~image_graph_dist_thr")
        self.proprio_graph_dist_thr = rospy.get_param("~proprio_graph_dist_thr")
        self.network_input_image_height = rospy.get_param("~network_input_image_height")
        self.network_input_image_width = rospy.get_param("~network_input_image_width")
        self.segmentation_type = rospy.get_param("~segmentation_type")
        self.feature_type = rospy.get_param("~feature_type")
        self.dino_patch_size = rospy.get_param("~dino_patch_size")
        self.confidence_std_factor = rospy.get_param("~confidence_std_factor")
        self.scale_traversability = rospy.get_param("~scale_traversability")
        self.scale_traversability_max_fpr = rospy.get_param("~scale_traversability_max_fpr")
        self.min_samples_for_training = rospy.get_param("~min_samples_for_training")
        self.vis_node_index = rospy.get_param("~debug_supervision_node_index_from_last")

        # Supervision Generator
        self.robot_max_velocity = rospy.get_param("~robot_max_velocity")
        self.untraversable_thr = rospy.get_param("~untraversable_thr")

        # Threads
        self.image_callback_rate = rospy.get_param("~image_callback_rate")  # hertz
        self.proprio_callback_rate = rospy.get_param("~proprio_callback_rate")  # hertz
        self.learning_thread_rate = rospy.get_param("~learning_thread_rate")  # hertz
        self.logging_thread_rate = rospy.get_param("~logging_thread_rate")  # hertz

        # Data storage
        self.mission_name = rospy.get_param("~mission_name")
        self.mission_timestamp = rospy.get_param("~mission_timestamp")

        # Print timings
        self.print_image_callback_time = rospy.get_param("~print_image_callback_time")
        self.print_proprio_callback_time = rospy.get_param("~print_proprio_callback_time")
        self.log_time = rospy.get_param("~log_time")
        self.log_confidence = rospy.get_param("~log_confidence")
        self.verbose = rospy.get_param("~verbose")

        # Select mode: # debug, online, extract_labels
        self.mode = WVNMode.from_string(rospy.get_param("~mode", "debug"))
        self.extraction_store_folder = rospy.get_param("~extraction_store_folder")
        self.vis_training_samples = rospy.get_param("~vis_training_samples")

        # Parse operation modes
        if self.mode == WVNMode.ONLINE:
            print("\nWARNING: online_mode enabled. The graph will not store any debug/training data such as images\n")

        elif self.mode == WVNMode.EXTRACT_LABELS:
            self.image_callback_rate = 3
            self.proprio_callback_rate = 4
            self.image_graph_dist_thr = 0.2
            self.proprio_graph_dist_thr = 0.1

            os.makedirs(os.path.join(self.extraction_store_folder, "image"), exist_ok=True)
            os.makedirs(os.path.join(self.extraction_store_folder, "supervision_mask"), exist_ok=True)

        # Experiment file
        exp_file = rospy.get_param("~exp")

        # Torch device
        self.device = rospy.get_param("~device")

        # Visualization
        self.colormap = rospy.get_param("~colormap")

        # Initialize traversability estimator parameters
        self.params = ExperimentParams()
        if exp_file != "nan":
            exp_override = load_yaml(os.path.join(WVN_ROOT_DIR, "cfg/exp", exp_file))
            self.params = override_params(self.params, exp_override)

        self.exp_cfg = dataclasses.asdict(self.params)

        self.params.general.name = self.mission_name
        self.params.general.timestamp = self.mission_timestamp
        self.params.general.log_confidence = self.log_confidence
        self.params.loss.confidence_std_factor = self.confidence_std_factor
        self.params.loss.w_temp = 0
        self.step = -1
        self.step_time = rospy.get_time()

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        if setup_fully:
            # Initialize TF listener
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            # Robot state callback
            robot_state_sub = message_filters.Subscriber(self.robot_state_topic, RobotState)
            cache1 = message_filters.Cache(robot_state_sub, 10)
            desired_twist_sub = message_filters.Subscriber(self.desired_twist_topic, TwistStamped)
            cache2 = message_filters.Cache(desired_twist_sub, 10)

            self.robot_state_sub = message_filters.ApproximateTimeSynchronizer(
                [robot_state_sub, desired_twist_sub], queue_size=10, slop=0.5
            )

            print(f"Start waiting for RobotState topic {self.robot_state_topic} being published!")
            rospy.wait_for_message(self.robot_state_topic, RobotState)
            print(f"Start waiting for TwistStamped topic {self.desired_twist_topic} being published!")
            rospy.wait_for_message(self.desired_twist_topic, TwistStamped)
            self.robot_state_sub.registerCallback(self.robot_state_callback)

            self.camera_handler = {}
            # Image callback
            for cam in self.camera_topics:
                # Initialize camera handler for given cam
                self.camera_handler[cam] = {}
                # Store camera name
                self.camera_topics[cam]["name"] = cam

                # Set subscribers
                imagefeat_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures)
                info_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo)
                sync = message_filters.ApproximateTimeSynchronizer([imagefeat_sub, info_sub], queue_size=4, slop=0.5)
                sync.registerCallback(self.imagefeat_callback, self.camera_topics[cam])

                self.camera_handler[cam]["image_sub"] = imagefeat_sub
                self.camera_handler[cam]["info_sub"] = info_sub
                self.camera_handler[cam]["sync"] = sync

        # 3D outputs
        self.pub_debug_proprio_graph = rospy.Publisher(
            "/wild_visual_navigation_node/proprioceptive_graph", Path, queue_size=10
        )
        self.pub_mission_graph = rospy.Publisher("/wild_visual_navigation_node/mission_graph", Path, queue_size=10)
        self.pub_graph_footprints = rospy.Publisher(
            "/wild_visual_navigation_node/graph_footprints", Marker, queue_size=10
        )
        # 1D outputs
        self.pub_instant_traversability = rospy.Publisher(
            "/wild_visual_navigation_node/instant_traversability", Float32, queue_size=10
        )
        self.pub_system_state = rospy.Publisher("/wild_visual_navigation_node/system_state", SystemState, queue_size=10)

        # Services
        # Like, reset graph or the like
        self.save_checkpt_service = rospy.Service("~save_checkpoint", SaveCheckpoint, self.save_checkpoint_callback)
        self.load_checkpt_service = rospy.Service("~load_checkpoint", LoadCheckpoint, self.load_checkpoint_callback)

        self.pause_learning_service = rospy.Service("~pause_learning", SetBool, self.pause_learning_callback)
        self.reset_service = rospy.Service("~reset", Trigger, self.reset_callback)

    def pause_learning_callback(self, req):
        """Start and stop the network training"""
        prev_state = self.traversability_estimator.pause_learning
        self.traversability_estimator.pause_learning = req.data
        if not req.data and prev_state:
            message = "Resume training!"
        elif req.data and prev_state:
            message = "Training was already paused!"
        elif not req.data and not prev_state:
            message = "Training was already running!"
        elif req.data and not prev_state:
            message = "Pause training!"
        message += f" Updated the network for {self.traversability_estimator.step} steps"

        return True, message

    def reset_callback(self, req):
        """Resets the system"""
        print("WARNING: System reset!")

        print("Storing learned checkpoint...", end="")
        self.traversability_estimator.save_checkpoint(self.params.general.model_path, "last_checkpoint.pt")
        print("done")

        if self.log_time:
            print("Storing timer data...", end="")
            self.timer.store(folder=self.params.general.model_path)
            print("done")

        # Create new mission folder
        create_experiment_folder(self.params, load_env())

        # Reset traversability estimator
        self.traversability_estimator.reset()

        print("Reset done")
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
            message = f"[WARNING] Store checkpoint {req.checkpoint_name} default mission path: {self.params.general.model_path}/{req.checkpoint_name}"
            req.mission_path = self.params.general.model_path
        else:
            message = f"Store checkpoint {req.checkpoint_name} to: {req.mission_path}/{req.checkpoint_name}"

        self.traversability_estimator.save_checkpoint(req.mission_path, req.checkpoint_name)
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
        self.traversability_estimator.load_checkpoint(checkpoint_path)
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
            trans = (res.transform.translation.x, res.transform.translation.y, res.transform.translation.z)
            rot = np.array(
                [res.transform.rotation.x, res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w]
            )
            rot /= np.linalg.norm(rot)
            return (trans, tuple(rot))
        except Exception as e:
            if self.verbose:
                print("Error in query tf: ", e)
                rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)

    @accumulate_time
    def robot_state_callback(self, state_msg, desired_twist_msg: TwistStamped):
        """Main callback to process proprioceptive info (robot state)

        Args:
            state_msg (wild_visual_navigation_msgs/RobotState): Robot state message
            desired_twist_msg (geometry_msgs/TwistStamped): Desired twist message
        """
        self.system_events["robot_state_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        try:
            ts = state_msg.header.stamp.to_sec()
            if abs(ts - self.last_proprio_ts) < 1.0 / self.proprio_callback_rate:
                self.system_events["robot_state_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to rate",
                }
                return
            self.last_propio_ts = ts

            # Query transforms from TF
            suc, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(self.fixed_frame, self.base_frame, state_msg.header.stamp), device=self.device
            )
            if not suc:
                self.system_events["robot_state_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_base_in_world",
                }
                return

            suc, pose_footprint_in_base = rc.ros_tf_to_torch(
                self.query_tf(self.base_frame, self.footprint_frame, state_msg.header.stamp), device=self.device
            )
            if not suc:
                self.system_events["robot_state_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_footprint_in_base",
                }
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
                desired_twist_in_base=desired_twist_tensor,
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

            if self.mode == WVNMode.DEBUG or self.mode == WVNMode.ONLINE:
                self.visualize_proprioception()

            if self.print_proprio_callback_time:
                print(self.timer)

            self.system_events["robot_state_callback_state"] = {
                "time": rospy.get_time(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)
            self.system_events["robot_state_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }

            raise Exception("Error in robot state callback")

    @accumulate_time
    def imagefeat_callback(self, imagefeat_msg: ImageFeatures, info_msg: CameraInfo, camera_options: dict):
        """Main callback to process incoming images

        Args:
            imagefeat_msg (wild_visual_navigation_msg/ImageFeatures): Incoming imagefeatures
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
        """
        self.system_events["image_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        if self.verbose:
            print(f"\nImage callback: {camera_options['name']}... ", end="")
        try:

            # Run the callback so as to match the desired rate
            ts = imagefeat_msg.header.stamp.to_sec()
            if abs(ts - self.last_image_ts) < 1.0 / self.image_callback_rate:
                if self.verbose:
                    print("skip")
                return
            else:
                if self.verbose:
                    print("process")
            self.last_image_ts = ts

            # Query transforms from TF
            suc, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(self.fixed_frame, self.base_frame, imagefeat_msg.header.stamp), device=self.device
            )
            if not suc:
                self.system_events["image_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_base_in_world",
                }
                return
            suc, pose_cam_in_base = rc.ros_tf_to_torch(
                self.query_tf(self.base_frame, imagefeat_msg.header.frame_id, imagefeat_msg.header.stamp),
                device=self.device,
            )

            if not suc:
                self.system_events["image_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_cam_in_base",
                }
                return
            # Prepare image projector
            K, H, W = rc.ros_cam_info_to_tensors(info_msg, device=self.device)
            image_projector = ImageProjector(
                K=K, h=H, w=W, new_h=self.network_input_image_height, new_w=self.network_input_image_width
            )

            # Add image to base node
            # convert image message to torch image
            feature_segments = rc.ros_image_to_torch(
                imagefeat_msg.feature_segments, desired_encoding="passthrough", device=self.device
            ).clone()
            h_small, w_small = feature_segments.shape[1:3]
            # Create mission node for the graph
            mission_node = MissionNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                pose_cam_in_base=pose_cam_in_base,
                image=torch.zeros((3, h_small, w_small), device=self.device, dtype=torch.float32),
                image_projector=image_projector,
                camera_name=camera_options["name"],
                use_for_training=camera_options["use_for_training"],
            )
            ma = imagefeat_msg.features
            dims = tuple(map(lambda x: x.size, ma.layout.dim))
            mission_node.features = torch.from_numpy(
                np.array(ma.data, dtype=float).reshape(dims).astype(np.float32)
            ).to(self.device)
            mission_node.feature_segments = feature_segments[0]

            # Add node to graph
            added_new_node = self.traversability_estimator.add_mission_node(mission_node, update_features=False)

            if self.mode == WVNMode.DEBUG:
                # Publish current predictions
                self.visualize_mission_graph()
                # Publish supervision data depending on the mode
                self.visualize_debug()

            # Print callback time if required
            if self.print_image_callback_time:
                print(self.timer)

            self.system_events["image_callback_state"] = {"time": rospy.get_time(), "value": "executed successfully"}

        except Exception as e:
            traceback.print_exc()
            print("error image callback", e)
            self.system_events["image_callback_state"] = {"time": rospy.get_time(), "value": f"failed to execute {e}"}
            raise Exception("Error in image callback")

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
            if self.verbose:
                print(f"number of points for footprint is {len(footprints_marker.points)}")
            return
        self.pub_graph_footprints.publish(footprints_marker)
        self.pub_debug_proprio_graph.publish(proprio_graph_msg)

        # Publish latest traversability
        self.pub_instant_traversability.publish(self.supervision_generator.traversability)
        self.system_events["visualize_proprioception"] = {"time": rospy.get_time(), "value": f"executed successfully"}

    @accumulate_time
    def visualize_mission_graph(self):
        """Publishes all the visualizations related to the mission graph"""
        # Get current time for later
        now = rospy.Time.now()

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
    node_name = "wvn_learning_node"
    rospy.init_node(node_name)
    if rospy.get_param("~reload_default_params", True):
        import rospkg

        rospack = rospkg.RosPack()
        wvn_path = rospack.get_path("wild_visual_navigation_ros")
        os.system(f"rosparam load {wvn_path}/config/wild_visual_navigation/default.yaml wvn_learning_node")
        os.system(
            f"rosparam load {wvn_path}/config/wild_visual_navigation/inputs/wide_angle_front_compressed.yaml wvn_learning_node"
        )

    wvn = WvnLearning()
    rospy.spin()
