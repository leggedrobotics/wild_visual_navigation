from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import ImageFeatures
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation.model import get_model
from wild_visual_navigation.utils import ConfidenceGenerator
from wild_visual_navigation.utils import AnomalyLoss

import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import MultiArrayDimension

import os
import torch
import numpy as np
from omegaconf import OmegaConf, read_write
from torch_geometric.data import Data
import torch.nn.functional as F
from threading import Thread, Event
from prettytable import PrettyTable
from termcolor import colored
import signal
import sys


class WvnFeatureExtractor:
    def __init__(self):
        # Read params
        self.read_params()

        self.feature_extractor = FeatureExtractor(
            self.ros_params.device,
            segmentation_type=self.ros_params.segmentation_type,
            feature_type=self.ros_params.feature_type,
            input_size=self.ros_params.network_input_image_height,
            slic_num_components=self.ros_params.slic_num_components,
            dino_dim=self.ros_params.dino_dim,
        )
        self.i = 0

        self.model = get_model(self.params.model).to(self.ros_params.device)
        self.model.eval()

        if not self.anomaly_detection:
            self.confidence_generator = ConfidenceGenerator(
                method=self.params.loss.method,
                std_factor=self.params.loss.confidence_std_factor,
                anomaly_detection=self.anomaly_detection,
            )
            self.ros_params.scale_traversability = True
        else:
            self.traversability_loss = AnomalyLoss(
                **self.params.loss_anomaly,
                log_enabled=self.params.general.log_confidence,
                log_folder=self.params.general.model_path,
            )
            self.traversability_loss.to(self.ros_params.device)
            self.ros_params.scale_traversability = False

        if self.ros_params.verbose:
            self.log_data = {}
            self.status_thread_stop_event = Event()
            self.status_thread = Thread(target=self.status_thread_loop, name="status")
            self.run_status_thread = True
            self.status_thread.start()

        self.setup_ros()

        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self, *args, **kwargs):
        self.run_status_thread = False
        self.status_thread_stop_event.set()
        self.status_thread.join()

        rospy.signal_shutdown(f"Wild Visual Navigation Feature Extraction killed {args}")
        sys.exit(0)

    def status_thread_loop(self):
        rate = rospy.Rate(self.ros_params.status_thread_rate)
        # Learning loop
        while self.run_status_thread:
            self.status_thread_stop_event.wait(timeout=0.01)
            if self.status_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]

            for k, v in self.log_data.items():
                if "time" in k:
                    d = t - v
                    if d < 0:
                        c = "red"
                    if d < 0.2:
                        c = "green"
                    elif d < 1.0:
                        c = "yellow"
                    else:
                        c = "red"
                    x.add_row([k, colored(round(d, 2), c)])
                else:
                    x.add_row([k, v])
            print(x)
            # try:
            #    rate.sleep()
            # except Exception as e:
            #    rate = rospy.Rate(self.ros_params.status_thread_rate)
            #    print("Ignored jump pack in time!")
        self.status_thread_stop_event.clear()

    def read_params(self):
        """Reads all the parameters from the parameter server"""
        self.params = OmegaConf.structured(ExperimentParams)
        self.ros_params = OmegaConf.structured(RosFeatureExtractorNodeParams)

        # Override the empty dataclass with values from rosparm server
        with read_write(self.ros_params):
            for k in self.ros_params.keys():
                self.ros_params[k] = rospy.get_param(f"~{k}")

        with read_write(self.params):
            self.params.loss.confidence_std_factor = self.ros_params.confidence_std_factor

        self.anomaly_detection = self.params.model.name == "LinearRnvp"

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        # Image callback

        self.camera_handler = {}

        if self.ros_params.verbose:
            # DEBUG Logging
            self.log_data[f"time_last_model"] = -1
            self.log_data[f"nr_model_updates"] = -1

        for cam in self.ros_params.camera_topics:
            if self.ros_params.verbose:
                # DEBUG Logging
                self.log_data[f"nr_images_{cam}"] = 0
                self.log_data[f"time_last_image_{cam}"] = -1

            # Initialize camera handler for given cam
            self.camera_handler[cam] = {}
            # Store camera name
            self.ros_params.camera_topics[cam]["name"] = cam

            # Camera info
            camera_info_msg = rospy.wait_for_message(
                self.ros_params.camera_topics[cam]["info_topic"], CameraInfo, timeout=15
            )
            K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self.ros_params.device)

            self.camera_handler[cam]["camera_info"] = camera_info_msg
            self.camera_handler[cam]["K"] = K
            self.camera_handler[cam]["H"] = H
            self.camera_handler[cam]["W"] = W

            image_projector = ImageProjector(
                K=self.camera_handler[cam]["K"],
                h=self.camera_handler[cam]["H"],
                w=self.camera_handler[cam]["W"],
                new_h=self.ros_params.network_input_image_height,
                new_w=self.ros_params.network_input_image_width,
            )
            msg = self.camera_handler[cam]["camera_info"]
            msg.width = self.ros_params.network_input_image_width
            msg.height = self.ros_params.network_input_image_height
            msg.K = image_projector.scaled_camera_matrix[0, :3, :3].cpu().numpy().flatten().tolist()
            msg.P = image_projector.scaled_camera_matrix[0, :3, :4].cpu().numpy().flatten().tolist()

            with read_write(self.ros_params):
                self.camera_handler[cam]["camera_info_msg_out"] = msg
                self.camera_handler[cam]["image_projector"] = image_projector

            # Set subscribers
            base_topic = self.ros_params.camera_topics[cam]["image_topic"].replace("/compressed", "")
            is_compressed = self.ros_params.camera_topics[cam]["image_topic"] != base_topic
            if is_compressed:
                # TODO study the effect of the buffer size
                image_sub = rospy.Subscriber(
                    self.ros_params.camera_topics[cam]["image_topic"],
                    CompressedImage,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            else:
                image_sub = rospy.Subscriber(
                    self.ros_params.camera_topics[cam]["image_topic"],
                    Image,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            self.camera_handler[cam]["image_sub"] = image_sub

            # Set publishers
            trav_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/traversability", Image, queue_size=10)
            info_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo, queue_size=10)
            self.camera_handler[cam]["trav_pub"] = trav_pub
            self.camera_handler[cam]["info_pub"] = info_pub
            if self.anomaly_detection and self.ros_params.camera_topics[cam]["publish_confidence"]:
                print(colored("Warning force set public confidence to false", "red"))
                self.ros_params.camera_topics[cam]["publish_confidence"] = False

            if self.ros_params.camera_topics[cam]["publish_input_image"]:
                input_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/image_input", Image, queue_size=10)
                self.camera_handler[cam]["input_pub"] = input_pub

            if self.ros_params.camera_topics[cam]["publish_confidence"]:
                conf_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/confidence", Image, queue_size=10)
                self.camera_handler[cam]["conf_pub"] = conf_pub

            if self.ros_params.camera_topics[cam]["use_for_training"]:
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
        if self.ros_params.verbose:
            # DEBUG Logging
            self.log_data[f"nr_images_{cam}"] += 1
            self.log_data[f"time_last_image_{cam}"] = rospy.get_time()

        # Update model from file if possible
        self.load_model()
        # Convert image message to torch image
        torch_image = rc.ros_image_to_torch(image_msg, device=self.ros_params.device)
        torch_image = self.camera_handler[cam]["image_projector"].resize_image(torch_image)
        C, H, W = torch_image.shape

        _, feat, seg, center, dense_feat = self.feature_extractor.extract(
            img=torch_image[None],
            return_centers=False,
            return_dense_features=True,
            n_random_pixels=100,
        )

        if self.ros_params.prediction_per_pixel:
            # Evaluate traversability
            data = Data(x=dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1]))
        else:
            # input_feat = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
            input_feat = feat[seg.reshape(-1)]
            data = Data(x=input_feat)

        # Evaluate traversability
        prediction = self.model.forward(data)

        if not self.anomaly_detection:
            out_trav = prediction.reshape(H, W, -1)[:, :, 0]

            # Publish traversability
            if self.ros_params.scale_traversability:
                # Apply piecewise linear scaling 0->0; threshold->0.5; 1->1
                traversability = out_trav.clone()
                m = traversability < self.ros_params.traversability_threshold
                # Scale untraversable
                traversability[m] *= 0.5 / self.ros_params.traversability_threshold
                # Scale traversable
                traversability[~m] -= self.ros_params.traversability_threshold
                traversability[~m] *= 0.5 / (1 - self.ros_params.traversability_threshold)
                traversability[~m] += 0.5
                traversability = traversability.clip(0, 1)
                # TODO Check if this was a bug
                out_trav = traversability
        else:
            loss, loss_aux, trav = self.traversability_loss(None, prediction)

            out_trav = trav.reshape(H, W, -1)[:, :, 0]

            # Clip to binary output
            if self.ros_params.clip_to_binary:
                out_trav = torch.where(out_trav.squeeze() <= self.ros_params.traversability_threshold, 0.0, 1.0)

        msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
        msg.header = image_msg.header
        msg.width = out_trav.shape[0]
        msg.height = out_trav.shape[1]
        self.camera_handler[cam]["trav_pub"].publish(msg)

        msg = self.camera_handler[cam]["camera_info_msg_out"]
        msg.header = image_msg.header
        self.camera_handler[cam]["info_pub"].publish(msg)

        # Publish image
        if self.ros_params.camera_topics[cam]["publish_input_image"]:
            msg = rc.numpy_to_ros_image((torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8), "rgb8")
            msg.header = image_msg.header
            msg.width = torch_image.shape[1]
            msg.height = torch_image.shape[2]
            self.camera_handler[cam]["input_pub"].publish(msg)

        # Publish confidence
        if self.ros_params.camera_topics[cam]["publish_confidence"]:
            loss_reco = F.mse_loss(prediction[:, 1:], data.x, reduction="none").mean(dim=1)
            confidence = self.confidence_generator.inference_without_update(x=loss_reco)
            out_confidence = confidence.reshape(H, W)
            msg = rc.numpy_to_ros_image(out_confidence.cpu().numpy(), "passthrough")
            msg.header = image_msg.header
            msg.width = out_confidence.shape[0]
            msg.height = out_confidence.shape[1]
            self.camera_handler[cam]["conf_pub"].publish(msg)

        # Publish features and feature_segments
        if self.ros_params.camera_topics[cam]["use_for_training"]:
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
                k = list(self.model.state_dict().keys())[-1]

                if (self.model.state_dict()[k] != res[k]).any():
                    if self.ros_params.verbose:
                        self.log_data[f"time_last_model"] = rospy.get_time()
                        self.log_data[f"nr_model_updates"] += 1

                self.model.load_state_dict(res, strict=False)

                try:
                    if res["traversability_threshold"] is not None:
                        # TODO Verify if this works or the writing is need
                        self.ros_params.traversability_threshold = res["traversability_threshold"]
                    if res["confidence_generator"] is not None:
                        self.confidence_generator_state = res["confidence_generator"]

                    self.confidence_generator_state = res["confidence_generator"]
                    self.confidence_generator.var = self.confidence_generator_state["var"]
                    self.confidence_generator.mean = self.confidence_generator_state["mean"]
                    self.confidence_generator.std = self.confidence_generator_state["std"]
                except:
                    pass

        except Exception as e:
            if self.ros_params.verbose:
                print(f"Model Loading Failed: {e}")


if __name__ == "__main__":
    node_name = "wvn_feature_extractor_node"
    rospy.init_node(node_name)

    if rospy.get_param("~reload_default_params", True):
        import rospkg

        rospack = rospkg.RosPack()
        wvn_path = rospack.get_path("wild_visual_navigation_ros")
        os.system(f"rosparam load {wvn_path}/config/wild_visual_navigation/default.yaml wvn_feature_extractor_node")
        os.system(
            f"rosparam load {wvn_path}/config/wild_visual_navigation/inputs/alphasense_compressed_front.yaml wvn_feature_extractor_node"
        )

    wvn = WvnFeatureExtractor()
    rospy.spin()
