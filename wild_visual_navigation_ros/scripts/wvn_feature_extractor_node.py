#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import ImageFeatures
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation_ros.scheduler import Scheduler
from wild_visual_navigation_ros.reload_rosparams import reload_rosparams
from wild_visual_navigation.model import get_model
from wild_visual_navigation.utils import ConfidenceGenerator
from wild_visual_navigation.utils import AnomalyLoss

import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import MultiArrayDimension

import torch
import numpy as np
import torch.nn.functional as F
import signal
import sys
import traceback
from omegaconf import OmegaConf, read_write
from wild_visual_navigation.utils import Data
from os.path import join
from threading import Thread, Event
from prettytable import PrettyTable
from termcolor import colored
import os


class WvnFeatureExtractor:
    def __init__(self, node_name):
        # Read params
        self.read_params()

        # Initialize variables
        self._node_name = node_name
        self._load_model_counter = 0

        # Timers to control the rate of the subscriber
        self._last_checkpoint_ts = rospy.get_time()

        # Setup modules
        self._feature_extractor = FeatureExtractor(
            self._ros_params.device,
            segmentation_type=self._ros_params.segmentation_type,
            feature_type=self._ros_params.feature_type,
            patch_size=self._ros_params.dino_patch_size,
            backbone_type=self._ros_params.dino_backbone,
            input_size=self._ros_params.network_input_image_height,
            slic_num_components=self._ros_params.slic_num_components,
        )

        # Load model
        # We manually update the input size to the models depending on the chosen features
        self._params.model.simple_mlp_cfg.input_size = self._feature_extractor.feature_dim
        self._params.model.double_mlp_cfg.input_size = self._feature_extractor.feature_dim
        self._params.model.simple_gcn_cfg.input_size = self._feature_extractor.feature_dim
        self._params.model.linear_rnvp_cfg.input_size = self._feature_extractor.feature_dim
        self._model = get_model(self._params.model).to(self._ros_params.device)
        self._model.eval()

        if self.anomaly_detection:
            self._confidence_generator = ConfidenceGenerator(
                method=self._params.loss_anomaly.method, std_factor=self._params.loss_anomaly.confidence_std_factor
            )

        else:
            self._confidence_generator = ConfidenceGenerator(
                method=self._params.loss.method, std_factor=self._params.loss.confidence_std_factor
            )
        self._log_data = {}
        self.setup_ros()

        # Setup verbosity levels
        if self._ros_params.verbose:

            self._status_thread_stop_event = Event()
            self._status_thread = Thread(target=self.status_thread_loop, name="status")
            self._run_status_thread = True
            self._status_thread.start()

        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self, *args, **kwargs):
        self._run_status_thread = False
        self._status_thread_stop_event.set()
        self._status_thread.join()

        rospy.signal_shutdown(f"Wild Visual Navigation Feature Extraction killed {args}")
        sys.exit(0)

    def read_params(self):
        """Reads all the parameters from the parameter server"""
        self._params = OmegaConf.structured(ExperimentParams)
        self._ros_params = OmegaConf.structured(RosFeatureExtractorNodeParams)

        # Override the empty dataclass with values from rosparm server
        with read_write(self._ros_params):
            for k in self._ros_params.keys():
                self._ros_params[k] = rospy.get_param(f"~{k}")

        with read_write(self._params):
            self._params.loss.confidence_std_factor = self._ros_params.confidence_std_factor
            self._params.loss_anomaly.confidence_std_factor = self._ros_params.confidence_std_factor

        self.anomaly_detection = self._params.model.name == "LinearRnvp"

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        # Image callback

        self._camera_handler = {}
        self._camera_scheduler = Scheduler()

        if self._ros_params.verbose:
            # DEBUG Logging
            self._log_data[f"time_last_model"] = -1
            self._log_data[f"nr_model_updates"] = -1

        self._last_image_ts = {}

        for cam in self._ros_params.camera_topics:
            self._last_image_ts[cam] = rospy.get_time()
            if self._ros_params.verbose:
                # DEBUG Logging
                self._log_data[f"nr_images_{cam}"] = 0
                self._log_data[f"time_last_image_{cam}"] = -1

            # Initialize camera handler for given cam
            self._camera_handler[cam] = {}
            # Store camera name
            self._ros_params.camera_topics[cam]["name"] = cam

            # Add to scheduler
            self._camera_scheduler.add_process(cam, self._ros_params.camera_topics[cam]["scheduler_weight"])

            # Camera info
            t = self._ros_params.camera_topics[cam]["info_topic"]
            rospy.loginfo(f"[{self._node_name}] Waiting for camera info topic {t}")
            camera_info_msg = rospy.wait_for_message(self._ros_params.camera_topics[cam]["info_topic"], CameraInfo)
            rospy.loginfo(f"[{self._node_name}] Done")
            K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self._ros_params.device)

            self._camera_handler[cam]["camera_info"] = camera_info_msg
            self._camera_handler[cam]["K"] = K
            self._camera_handler[cam]["H"] = H
            self._camera_handler[cam]["W"] = W

            image_projector = ImageProjector(
                K=self._camera_handler[cam]["K"],
                h=self._camera_handler[cam]["H"],
                w=self._camera_handler[cam]["W"],
                new_h=self._ros_params.network_input_image_height,
                new_w=self._ros_params.network_input_image_width,
            )
            msg = self._camera_handler[cam]["camera_info"]
            msg.width = self._ros_params.network_input_image_width
            msg.height = self._ros_params.network_input_image_height
            msg.K = image_projector.scaled_camera_matrix[0, :3, :3].cpu().numpy().flatten().tolist()
            msg.P = image_projector.scaled_camera_matrix[0, :3, :4].cpu().numpy().flatten().tolist()

            with read_write(self._ros_params):
                self._camera_handler[cam]["camera_info_msg_out"] = msg
                self._camera_handler[cam]["image_projector"] = image_projector

            # Set subscribers
            base_topic = self._ros_params.camera_topics[cam]["image_topic"].replace("/compressed", "")
            is_compressed = self._ros_params.camera_topics[cam]["image_topic"] != base_topic
            if is_compressed:
                # TODO study the effect of the buffer size
                image_sub = rospy.Subscriber(
                    self._ros_params.camera_topics[cam]["image_topic"],
                    CompressedImage,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            else:
                image_sub = rospy.Subscriber(
                    self._ros_params.camera_topics[cam]["image_topic"],
                    Image,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            self._camera_handler[cam]["image_sub"] = image_sub

            # Set publishers
            trav_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/traversability",
                Image,
                queue_size=1,
            )
            info_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/camera_info",
                CameraInfo,
                queue_size=1,
            )
            self._camera_handler[cam]["trav_pub"] = trav_pub
            self._camera_handler[cam]["info_pub"] = info_pub
            if self.anomaly_detection and self._ros_params.camera_topics[cam]["publish_confidence"]:
                rospy.logwarn(f"[{self._node_name}] Warning force set public confidence to false")
                self._ros_params.camera_topics[cam]["publish_confidence"] = False

            if self._ros_params.camera_topics[cam]["publish_input_image"]:
                input_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/image_input",
                    Image,
                    queue_size=1,
                )
                self._camera_handler[cam]["input_pub"] = input_pub

            if self._ros_params.camera_topics[cam]["publish_confidence"]:
                conf_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/confidence",
                    Image,
                    queue_size=1,
                )
                self._camera_handler[cam]["conf_pub"] = conf_pub

            if self._ros_params.camera_topics[cam]["use_for_training"]:
                imagefeat_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/feat",
                    ImageFeatures,
                    queue_size=1,
                )
                self._camera_handler[cam]["imagefeat_pub"] = imagefeat_pub

    def status_thread_loop(self):
        rate = rospy.Rate(self._ros_params.status_thread_rate)
        # Learning loop
        while self._run_status_thread:
            self._status_thread_stop_event.wait(timeout=0.01)
            if self._status_thread_stop_event.is_set():
                rospy.logwarn(f"[{self._node_name}] Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]

            for k, v in self._log_data.items():
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
            print(f"[{self._node_name}]\n{x}")
            try:
                rate.sleep()
            except Exception:
                rate = rospy.Rate(self._ros_params.status_thread_rate)
                print(f"[{self._node_name}] Ignored jump pack in time!")
        self._status_thread_stop_event.clear()

    @torch.no_grad()
    def image_callback(self, image_msg: Image, cam: str):  # info_msg: CameraInfo
        """Main callback to process incoming images.

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
            cam (str): Camera name
        """
        # Check the rate
        ts = image_msg.header.stamp.to_sec()
        if abs(ts - self._last_image_ts[cam]) < 1.0 / self._ros_params.image_callback_rate:
            return

        # Check the scheduler
        if self._camera_scheduler.get() != cam:
            return
        else:
            if self._ros_params.verbose:
                rospy.loginfo(f"[{self._node_name}] Image callback: {cam} -> Process")

        self._last_image_ts[cam] = ts

        # If all the checks are passed, process the image
        try:
            if self._ros_params.verbose:
                # DEBUG Logging
                self._log_data[f"nr_images_{cam}"] += 1
                self._log_data[f"time_last_image_{cam}"] = rospy.get_time()

            # Update model from file if possible
            self.load_model(image_msg.header.stamp)

            # Convert image message to torch image
            torch_image = rc.ros_image_to_torch(image_msg, device=self._ros_params.device)
            torch_image = self._camera_handler[cam]["image_projector"].resize_image(torch_image)
            C, H, W = torch_image.shape

            # Extract features
            _, feat, seg, center, dense_feat = self._feature_extractor.extract(
                img=torch_image[None],
                return_centers=False,
                return_dense_features=True,
                n_random_pixels=100,
            )

            # Forward pass to predict traversability
            if self._ros_params.prediction_per_pixel:
                # Pixel-wise traversability prediction using the dense features
                data = Data(x=dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1]))
            else:
                # input_feat = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
                # Segment-wise traversability prediction using the average feature per segment
                input_feat = feat[seg.reshape(-1)]
                data = Data(x=input_feat)

            # Predict traversability per feature
            prediction = self._model.forward(data)

            if not self.anomaly_detection:
                out_trav = prediction.reshape(H, W, -1)[:, :, 0]
            else:
                losses = prediction["logprob"].sum(1) + prediction["log_det"]
                confidence = self._confidence_generator.inference_without_update(x=-losses)
                trav = confidence
                out_trav = trav.reshape(H, W, -1)[:, :, 0]

            msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
            msg.header = image_msg.header
            msg.width = out_trav.shape[0]
            msg.height = out_trav.shape[1]
            self._camera_handler[cam]["trav_pub"].publish(msg)

            msg = self._camera_handler[cam]["camera_info_msg_out"]
            msg.header = image_msg.header
            self._camera_handler[cam]["info_pub"].publish(msg)

            # Publish image
            if self._ros_params.camera_topics[cam]["publish_input_image"]:
                msg = rc.numpy_to_ros_image(
                    (torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8),
                    "rgb8",
                )
                msg.header = image_msg.header
                msg.width = torch_image.shape[1]
                msg.height = torch_image.shape[2]
                self._camera_handler[cam]["input_pub"].publish(msg)

            # Publish confidence
            if self._ros_params.camera_topics[cam]["publish_confidence"]:
                loss_reco = F.mse_loss(prediction[:, 1:], data.x, reduction="none").mean(dim=1)
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)
                out_confidence = confidence.reshape(H, W)
                msg = rc.numpy_to_ros_image(out_confidence.cpu().numpy(), "passthrough")
                msg.header = image_msg.header
                msg.width = out_confidence.shape[0]
                msg.height = out_confidence.shape[1]
                self._camera_handler[cam]["conf_pub"].publish(msg)

            # Publish features and feature_segments
            if self._ros_params.camera_topics[cam]["use_for_training"]:
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
                self._camera_handler[cam]["imagefeat_pub"].publish(msg)

        except Exception as e:
            traceback.print_exc()
            rospy.logerr(f"[self._node_name] error image callback", e)
            self.system_events["image_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
            raise Exception("Error in image callback")

        # Step scheduler
        self._camera_scheduler.step()

    def load_model(self, stamp):
        """Method to load the new model weights to perform inference on the incoming images

        Args:
            None
        """
        ts = stamp.to_sec()
        if abs(ts - self._last_checkpoint_ts) < 1.0 / self._ros_params.load_save_checkpoint_rate:
            return

        self._last_checkpoint_ts = ts

        # self._load_model_counter += 1
        # if self._load_model_counter % 10 == 0:
        p = join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
        # p = join(WVN_ROOT_DIR,"assets/checkpoints/mountain_bike_trail_fpr_0.25.pt")

        if os.path.exists(p):
            new_model_state_dict = torch.load(p)
            k = list(self._model.state_dict().keys())[-1]

            # check if the key is in state dict - this may be not the case if switched between models
            # assumption first key within state_dict is unique and sufficient to identify if a model has changed
            if k in new_model_state_dict:
                # check if the model has changed
                if (self._model.state_dict()[k] != new_model_state_dict[k]).any():
                    if self._ros_params.verbose:
                        self._log_data[f"time_last_model"] = rospy.get_time()
                        self._log_data[f"nr_model_updates"] += 1

                    self._model.load_state_dict(new_model_state_dict, strict=False)
                    if "confidence_generator" in new_model_state_dict.keys():
                        cg = new_model_state_dict["confidence_generator"]
                        self._confidence_generator.var = cg["var"]
                        self._confidence_generator.mean = cg["mean"]
                        self._confidence_generator.std = cg["std"]

                    if self._ros_params.verbose:
                        m, s, v = cg["mean"].item(), cg["std"].item(), cg["var"].item()
                        rospy.loginfo(f"[{self._node_name}] Loaded Confidence Generator {m}, std {s} var {v}")

        else:
            if self._ros_params.verbose:
                rospy.logerr(f"[{self._node_name}] Model Loading Failed")


if __name__ == "__main__":
    node_name = "wvn_feature_extractor_node"
    rospy.init_node(node_name)

    reload_rosparams(
        enabled=rospy.get_param("~reload_default_params", True),
        node_name=node_name,
        camera_cfg="wide_angle_dual",
    )

    wvn = WvnFeatureExtractor(node_name)
    rospy.spin()
