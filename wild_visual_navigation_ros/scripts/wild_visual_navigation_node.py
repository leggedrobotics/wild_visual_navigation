#!/usr/bin/python3
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import MissionNode, ProprioceptionNode
import wild_visual_navigation_ros.ros_converter as rc

from wild_visual_navigation_msgs.msg import RobotState
from wild_visual_navigation_msgs.srv import SaveLoadData, SaveLoadDataResponse
from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA, Float32
from std_srvs.srv import Trigger, TriggerResponse
from threading import Thread
from visualization_msgs.msg import Marker
import message_filters
import os
import rospy
import seaborn as sns
import tf
import torch

torch.cuda.empty_cache()


class WvnRosInterface:
    def __init__(self):
        self.last_ts = rospy.get_time()

        # Read params
        self.read_params()

        # Visualization
        self.color_palette = sns.color_palette(self.colormap, as_cmap=True)

        # Initialize traversability estimator
        self.traversability_estimator = TraversabilityEstimator(
            device=self.device,
            max_distance=self.traversability_radius,
            image_distance_thr=self.image_graph_dist_thr,
            proprio_distance_thr=self.proprio_graph_dist_thr,
            optical_flow_estimatior_type=self.optical_flow_estimatior_type,
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
        """Destructor
        Joins all the running threads
        """
        # Join threads
        if self.run_online_learning:
            self.learning_thread.join()

    def read_params(self):
        """Reads all the parameters from the parameter server"""
        # Topics
        self.robot_state_topic = rospy.get_param("~robot_state_topic", "/wild_visual_navigation_ros/robot_state")
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
        self.traversability_radius = rospy.get_param("~traversability_radius", 5.0)
        self.image_graph_dist_thr = rospy.get_param("~image_graph_dist_thr", 0.1)
        self.proprio_graph_dist_thr = rospy.get_param("~proprio_graph_dist_thr", 0.1)
        self.network_input_image_height = rospy.get_param("~network_input_image_height", 448)
        self.network_input_image_width = rospy.get_param("~network_input_image_width", 448)

        # Optical flow params
        self.optical_flow_estimatior_type = rospy.get_param("optical_flow_estimatior_type", "sparse")

        # Threads
        self.run_online_learning = rospy.get_param("~run_online_learning", True)
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

        # Visualization
        self.colormap = rospy.get_param("colormap", "RdYlBu")

    def setup_ros(self):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # Robot state callback
        robot_state_sub = message_filters.Subscriber(self.robot_state_topic, RobotState)
        desired_twist_sub = message_filters.Subscriber(self.desired_twist_topic, TwistStamped)
        self.robot_state_sub = message_filters.ApproximateTimeSynchronizer(
            [robot_state_sub, desired_twist_sub], queue_size=10, slop=0.1
        )
        self.robot_state_sub.registerCallback(self.robot_state_callback)

        # Image callback
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.info_sub = message_filters.Subscriber(self.info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub], 2, slop=0.1)
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
        self.pub_instant_traversability = rospy.Publisher(
            "/wild_visual_navigation_node/instant_traversability", Float32, queue_size=10
        )

        # Services
        # Like, reset graph or the like
        self.save_graph_service = rospy.Service("~save_graph", SaveLoadData, self.save_graph_callback)
        self.save_pickle_service = rospy.Service("~save_pickle", SaveLoadData, self.save_pickle_callback)
        self.save_checkpt_service = rospy.Service("~save_checkpoint", SaveLoadData, self.save_checkpoint_callback)
        self.load_checkpt_service = rospy.Service("~load_checkpoint", SaveLoadData, self.load_checkpoint_callback)

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

    def query_tf(self, parent_frame: str, child_frame: str):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
        """
        # Wait for required tfs
        self.tf_listener.waitForTransform(parent_frame, child_frame, rospy.Time(), rospy.Duration(1.0))

        try:
            (trans, rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            return (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)

    def robot_state_callback(self, state_msg, desired_twist_msg: TwistStamped):
        """Main callback to process proprioceptive info (robot state)

        Args:
            state_msg (wild_visual_navigation_msgs/RobotState): Robot state message
            desired_twist_msg (geometry_msgs/TwistStamped): Desired twist message
        """
        ts = state_msg.header.stamp.to_sec()

        # Query transforms from TF
        pose_base_in_world = rc.ros_tf_to_torch(self.query_tf(self.fixed_frame, self.base_frame), device=self.device)
        pose_footprint_in_base = rc.ros_tf_to_torch(
            self.query_tf(self.base_frame, self.footprint_frame), device=self.device
        )

        # The footprint requires a correction: we use the same orientation as the base
        pose_footprint_in_base[:3, :3] = torch.eye(3, device=self.device)

        # Convert state to tensor
        proprio_tensor, proprio_labels = rc.wvn_robot_state_to_torch(state_msg, device=self.device)
        twist_tensor = rc.twist_to_torch(state_msg.twist, linear="xy", angular=None, device=self.device)
        command_tensor = rc.twist_to_torch(desired_twist_msg, linear="xy", angular=None, device=self.device)

        # Update affordance
        affordance, affordance_var = affordance_generator.update_with_velocities(twist_tensor, command_tensor)

        # Create proprioceptive node for the graph
        proprio_node = ProprioceptionNode(
            timestamp=ts,
            pose_base_in_world=pose_base_in_world,
            pose_footprint_in_base=pose_footprint_in_base,
            width=self.robot_width,
            length=self.robot_length,
            height=self.robot_width,
            proprioception=proprio_tensor,
            traversability=affordance,
            traversability_var=affordance_var,
        )
        # Add node to graph
        self.traversability_estimator.add_proprio_node(proprio_node)

        # Visualizations
        self.visualize_proprioception()

    def image_callback(self, image_msg: Image, info_msg: CameraInfo):
        """Main callback to process incoming images

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
        """
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
        image_projector = ImageProjector(
            K=K, h=H, w=W, new_h=self.network_input_image_height, new_w=self.network_input_image_width
        )

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
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        # Set rate
        rate = rospy.Rate(self.learning_thread_rate)

        # Main loop
        while not rospy.is_shutdown():
            # Optimize model
            self.traversability_estimator.train()
            rate.sleep()

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
            c = ColorRGBA(r, g, b, 0.8)

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

        # Publish
        if len(footprints_marker.points) % 3 != 0:
            print(f"number of points for footprint is {len(footprints_marker.points)}")
            return
        self.pub_graph_footprints.publish(footprints_marker)
        self.pub_debug_proprio_graph.publish(proprio_graph_msg)

        # Instant traversability
        last_pnode = self.traversability_estimator.get_las

    def visualize_mission(self, mission_node: MissionNode = None):
        """Publishes all the visualizations related to mission graph, like the graph
        itself, visual features, supervision signals, and traversability estimates
        """
        # Get current time for later
        now = rospy.Time.now()

        # Publish predictions
        if mission_node is not None and self.run_online_learning:
            np_prediction_image, np_uncertainty_image = self.traversability_estimator.plot_mission_node_prediction(
                mission_node
            )
            self.pub_image_prediction.publish(rc.numpy_to_ros_image(np_prediction_image))
            self.pub_image_prediction_uncertainty.publish(rc.numpy_to_ros_image(np_uncertainty_image))

        # Publish reprojections of last node in graph
        # TODO: change visualization for a better node
        if len(self.traversability_estimator.get_mission_nodes()) > 0:
            nodes = self.traversability_estimator.get_mission_nodes()
            try:
                mission_node = nodes[-10]
            except Exception:
                mission_node = self.traversability_estimator.get_last_valid_mission_node()

            if mission_node is not None:
                # torch_mask = mission_node.supervision_mask
                # self.pub_image_mask.publish(rc.torch_to_ros_image(torch_mask))

                np_labeled_image, np_mask_image = self.traversability_estimator.plot_mission_node_training(mission_node)
                if np_labeled_image is None or np_mask_image is None:
                    return
                self.pub_image_labeled.publish(rc.numpy_to_ros_image(np_labeled_image))
                self.pub_image_mask.publish(rc.numpy_to_ros_image(np_mask_image))

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
