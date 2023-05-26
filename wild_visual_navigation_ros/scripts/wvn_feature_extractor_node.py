from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml, load_env, create_experiment_folder
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.utils import override_params
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import ImageFeatures
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation.learning.model import get_model
from wild_visual_navigation.utils import ConfidenceGenerator

import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Float32, MultiArrayDimension
from rospy.numpy_msg import numpy_msg
import message_filters

from pytictac import Timer, CpuTimer
import os
import torch
import numpy as np
import dataclasses
from torch_geometric.data import Data
import torch.nn.functional as F


class WvnFeatureExtractor:
    def __init__(self):
        # Read params
        self.read_params()
        self.feature_extractor = FeatureExtractor(
            self.device,
            segmentation_type=self.segmentation_type,
            feature_type=self.feature_type,
            input_size=self.network_input_image_height,
        )
        print("Segmentation type: ", self.segmentation_type)
        self.model = get_model(self.exp_cfg["model"]).to(self.device)
        self.model.eval()
        
        self.confidence_generator = ConfidenceGenerator(
            method=self.exp_cfg["loss"]["method"], std_factor=self.exp_cfg["loss"]["confidence_std_factor"]
        )
        self.scale_traversability = True
        self.traversability_thershold = 0.5

        self.i = 0
        
        self.setup_ros()

    def read_params(self):
        """Reads all the parameters from the parameter server"""
        self.device = rospy.get_param("~device")
        self.verbose = rospy.get_param("~verbose")

        # Topics
        self.camera_topics = rospy.get_param("~camera_topics")
        # Experiment file
        self.network_input_image_height = rospy.get_param("~network_input_image_height")
        self.network_input_image_width = rospy.get_param("~network_input_image_width")

        self.segmentation_type = rospy.get_param("~segmentation_type")
        self.feature_type = rospy.get_param("~feature_type")
        self.dino_patch_size = rospy.get_param("~dino_patch_size")
        self.publish_dense = rospy.get_param("~publish_dense")

        self.confidence_std_factor = rospy.get_param("~confidence_std_factor")
        self.scale_traversability = rospy.get_param("~scale_traversability")
        self.scale_traversability_max_fpr = rospy.get_param("~scale_traversability_max_fpr")

        # Initialize traversability estimator parameters
        # Experiment file
        exp_file = rospy.get_param("~exp")
        self.params = ExperimentParams()
        if exp_file != "nan":
            exp_override = load_yaml(os.path.join(WVN_ROOT_DIR, "cfg/exp", exp_file))
            self.params = override_params(self.params, exp_override)

        self.exp_cfg = dataclasses.asdict(self.params)
        self.exp_cfg["loss"]["confidence_std_factor"] = self.confidence_std_factor

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        # Image callback
        self.camera_handler = {}
        for cam in self.camera_topics:
            # Initialize camera handler for given cam
            self.camera_handler[cam] = {}
            # Store camera name
            self.camera_topics[cam]["name"] = cam

            # Camera info
            print("Get camera info from topic: ", self.camera_topics[cam]["info_topic"])
            try:
                camera_info_msg = rospy.wait_for_message(self.camera_topics[cam]["info_topic"], CameraInfo, timeout=5)
            except rospy.ROSException:
                rospy.logerr("Camera info topic not found: {}".format(self.camera_topics[cam]["info_topic"]))
            
            self.camera_topics[cam]["camera_info"] = camera_info_msg

            K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self.device)
            self.camera_topics[cam]["K"] = K
            self.camera_topics[cam]["H"] = H
            self.camera_topics[cam]["W"] = W

            image_projector = ImageProjector(
                K=self.camera_topics[cam]["K"],
                h=self.camera_topics[cam]["H"],
                w=self.camera_topics[cam]["W"],
                new_h=self.network_input_image_height,
                new_w=self.network_input_image_width,
            )
            msg = self.camera_topics[cam]["camera_info"]
            msg.width = self.network_input_image_width
            msg.height = self.network_input_image_height
            msg.K = image_projector.scaled_camera_matrix[0, :3, :3].cpu().numpy().flatten().tolist()
            msg.P = image_projector.scaled_camera_matrix[0, :3, :4].cpu().numpy().flatten().tolist()

            self.camera_topics[cam]["camera_info_msg_out"] = msg
            self.camera_topics[cam]["image_projector"] = image_projector

            # Set subscribers
            base_topic = self.camera_topics[cam]["image_topic"].replace("/compressed", "")
            is_compressed = self.camera_topics[cam]["image_topic"] != base_topic
            if is_compressed:
                # TODO study the effect of the buffer size
                image_sub = rospy.Subscriber(
                    self.camera_topics[cam]["image_topic"],
                    CompressedImage,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=5,
                )
            else:
                image_sub = rospy.Subscriber(
                    self.camera_topics[cam]["image_topic"], Image, self.image_callback, callback_args=cam, queue_size=5
                )
            self.camera_handler[cam]["image_sub"] = image_sub

            # Set publishers
            trav_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/traversability", Image, queue_size=10)
            info_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo, queue_size=10)
            self.camera_handler[cam]["trav_pub"] = trav_pub
            self.camera_handler[cam]["info_pub"] = info_pub

            if self.camera_topics[cam]["publish_input_image"]:
                input_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/image_input", Image, queue_size=10)
                self.camera_handler[cam]["input_pub"] = input_pub

            if self.camera_topics[cam]["publish_confidence"]:
                conf_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/confidence", Image, queue_size=10)
                self.camera_handler[cam]["conf_pub"] = conf_pub

            if self.camera_topics[cam]["use_for_training"]:
                imagefeat_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures, queue_size=10
                )
                self.camera_handler[cam]["imagefeat_pub"] = imagefeat_pub

    @torch.no_grad()
    def image_callback(self, image_msg: Image, cam: str):  #  info_msg: CameraInfo
        """Main callback to process incoming images.

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
            cam (str): Camera name
        """
        if self.verbose:
            print("Processing Camera: ", cam)

        # Update model from file if possible
        self.load_model()
        # Convert image message to torch image
        torch_image = rc.ros_image_to_torch(image_msg, device=self.device)
        torch_image = self.camera_topics[cam]["image_projector"].resize_image(torch_image)
        C, H, W = torch_image.shape

        _, feat, seg, center, dense_feat = self.feature_extractor.extract(
            img=torch_image[None],
            return_centers=False,
            return_dense_features=True,
            n_random_pixels=100,
        )

        if self.publish_dense:
            # Evaluate traversability
            data = Data(x=dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1]))
        else:
            input_feat = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
            input_feat = feat[seg.reshape(-1)]
            data = Data(x=input_feat)
            
        prediction = self.model.forward(data)
        out_trav = prediction.reshape(H, W, -1)[:, :, 0]
            

        # Publish traversability
        if self.scale_traversability:
            # Apply piecewise linear scaling 0->0; threshold->0.5; 1->1
            traversability = out_trav.clone()
            m = traversability < self.traversability_thershold
            # Scale untraversable
            traversability[m] *= 0.5 / self.traversability_thershold
            # Scale traversable
            traversability[~m] -= self.traversability_thershold
            traversability[~m] *= 0.5 / (1 - self.traversability_thershold)
            traversability[~m] += 0.5
            traversability = traversability.clip(0, 1)

        msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
        msg.header = image_msg.header
        msg.width = out_trav.shape[0]
        msg.height = out_trav.shape[1]
        self.camera_handler[cam]["trav_pub"].publish(msg)

        msg = self.camera_topics[cam]["camera_info_msg_out"]
        msg.header = image_msg.header
        self.camera_handler[cam]["info_pub"].publish(msg)

        # Publish image
        if self.camera_topics[cam]["publish_input_image"]:
            msg = rc.numpy_to_ros_image((torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8), "rgb8")
            msg.header = image_msg.header
            msg.width = torch_image.shape[1]
            msg.height = torch_image.shape[2]
            self.camera_handler[cam]["input_pub"].publish(msg)

        # Publish confidence
        if self.camera_topics[cam]["publish_confidence"]:
            loss_reco = F.mse_loss(prediction[:, 1:], data.x, reduction="none").mean(dim=1)
            confidence = self.confidence_generator.inference_without_update(x=loss_reco)
            out_confidence = confidence.reshape(H, W)
            
            out_confidence = out_confidence.clip(0, 1)
            
            msg = rc.numpy_to_ros_image(out_confidence.cpu().numpy(), "passthrough")
            msg.header = image_msg.header
            msg.width = out_confidence.shape[0]
            msg.height = out_confidence.shape[1]
            self.camera_handler[cam]["conf_pub"].publish(msg)

        # Publish features and feature_segments
        if self.camera_topics[cam]["use_for_training"]:
            msg = ImageFeatures()
            msg.header = image_msg.header
            msg.feature_segments = rc.numpy_to_ros_image(seg.cpu().numpy().astype(np.int32), "passthrough")
            msg.feature_segments.header = image_msg.header
            feat_np = feat.cpu().numpy()

            mad1 = MultiArrayDimension()
            mad1.label = "n"
            mad1.size = feat_np.shape[0]
            mad1.stride = feat_np.shape[0] * feat_np.shape[1]

            mad2 = MultiArrayDimension()
            mad2.label = "feat"
            mad2.size = feat_np.shape[1]
            mad2.stride = feat_np.shape[1]

            msg.features.data = feat_np.flatten().tolist()
            msg.features.layout.dim.append(mad1)
            msg.features.layout.dim.append(mad2)
            self.camera_handler[cam]["imagefeat_pub"].publish(msg)

    def load_model(self):
        try:
            self.i += 1
            if self.i % 100 == 0:
                res = torch.load(f"{WVN_ROOT_DIR}/tmp_state_dict2.pt")
                if (self.model.state_dict()["layers.0.weight"] != res["layers.0.weight"]).any():
                    if self.verbose:
                        print("Model updated.")
                    self.model.load_state_dict(res, strict=False)
                    self.traversability_thershold = res["traversability_thershold"]
                    self.confidence_generator_state = res["confidence_generator"]

                    self.confidence_generator.var = self.confidence_generator_state["var"]
                    self.confidence_generator.mean = self.confidence_generator_state["mean"]
                    self.confidence_generator.std = self.confidence_generator_state["std"]
                else:
                    if self.verbose:
                        print("Model did not change.")
        except Exception as e:
            if self.verbose:
                print(f"Model Loading Failed: {e}")


if __name__ == "__main__":
    node_name = "wvn_feature_extractor_node"
    os.system(f"rosparam delete {node_name}")
    os.system(
        f"rosparam load /home/jonfrey/git/wild_visual_navigation/wild_visual_navigation_ros/config/wild_visual_navigation/default.yaml {node_name}"
    )
    os.system(
        f"rosparam load /home/jonfrey/git/wild_visual_navigation/wild_visual_navigation_ros/config/wild_visual_navigation/inputs/alphasense_single.yaml {node_name}"
    )
    rospy.init_node(node_name)
    wvn = WvnFeatureExtractor()
    rospy.spin()


# rosbag record /alphasense_driver_ros/cam4/color_rect/camera_info \
# /alphasense_driver_ros/cam4/color_rect/image \
# /anymal_low_level_controller/actuator_readings \
# /bpearl_front/point_cloud \
# /bpearl_rear/point_cloud \
# /clicked_point \
# /clock \
# /elevation_mapping/elevation_map_raw \
# /elevation_mapping/semantic_map_raw \
# /graph_msf/accel_bias \
# /graph_msf/est_odometry_odom_imu \
# /graph_msf/est_odometry_world_imu \
# /graph_msf/est_path_odom_imu \
# /graph_msf/est_path_world_imu \
# /graph_msf/gyro_bias \
# /graph_msf/measGnss_path_map_gnss \
# /graph_msf/measLiDAR_path_map_imu \
# /graph_msf/opt_odometry_world_imu \
# /graph_msf/opt_path_world_imu \
# /image_mask_filter/bpearl_front/point_cloud \
# /image_mask_filter/bpearl_rear/point_cloud \
# /initialpose \
# /lidar/packets \
# /lidar/point_cloud \
# /lidar_nodelet_cloud/parameter_descriptions \
# /lidar_nodelet_cloud/parameter_updates \
# /lidar_nodelet_manager/bond \
# /log/state/desiredRobotTwist \
# /move_base_simple/goal \
# /msf_compslam_lio_body_imu/msf_core/odometry \
# /robot_self_filter/bpearl_front/point_cloud \
# /robot_self_filter/bpearl_rear/point_cloud \
# /rosout \
# /rosout_agg \
# /sensors/battery_voltage \
# /sensors/imu \
# /stack_launcher/node_states \
# /state_estimator/anymal_state \
# /state_estimator/odometry \
# /state_estimator/pose_in_odom \
# /tf \
# /tf_static \
# /twist_mux/twist \
# /wild_visual_navigation_node/front/camera_info \
# /wild_visual_navigation_node/front/confidence \
# /wild_visual_navigation_node/front/debug/last_image_confidence \
# /wild_visual_navigation_node/front/debug/last_image_traversability \
# /wild_visual_navigation_node/front/debug/last_node_image_labeled \
# /wild_visual_navigation_node/front/debug/last_node_image_mask \
# /wild_visual_navigation_node/front/feat \
# /wild_visual_navigation_node/front/image_input \
# /wild_visual_navigation_node/front/traversability \
# /wild_visual_navigation_node/graph_footprints \
# /wild_visual_navigation_node/graph_footprints_array \
# /wild_visual_navigation_node/instant_traversability \
# /wild_visual_navigation_node/left/camera_info \
# /wild_visual_navigation_node/left/traversability \
# /wild_visual_navigation_node/mission_graph \
# /wild_visual_navigation_node/proprioceptive_graph \
# /wild_visual_navigation_node/right/camera_info \
# /wild_visual_navigation_node/right/traversability \
# /wild_visual_navigation_node/robot_state \
# /wild_visual_navigation_node/system_state \
# /wild_visual_navigation_visu_0/traversability_overlayed \
# /wild_visual_navigation_visu_1/confidence_overlayed
