from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml, load_env, create_experiment_folder
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.utils import override_params
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import ImageFeatures
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation.learning.model import get_model


import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Float32, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

from pytictac import Timer
import os
import torch
import numpy as np
import dataclasses
from torch_geometric.data import Data


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
        self.setup_ros()

        self.exp_cfg = dataclasses.asdict(self.params)
        self.model = get_model(self.exp_cfg["model"]).to(self.device)
        self.model.eval()

    def read_params(self):
        """Reads all the parameters from the parameter server"""
        # Topics
        self.camera_topics = rospy.get_param("~camera_topics")
        # Experiment file
        self.network_input_image_height = rospy.get_param("~network_input_image_height")
        self.network_input_image_width = rospy.get_param("~network_input_image_width")

        self.segmentation_type = rospy.get_param("~segmentation_type")
        self.feature_type = rospy.get_param("~feature_type")
        self.dino_patch_size = rospy.get_param("~dino_patch_size")

        # Initialize traversability estimator parameters
        # Experiment file
        exp_file = rospy.get_param("~exp")
        self.params = ExperimentParams()
        if exp_file != "nan":
            exp_override = load_yaml(os.path.join(WVN_ROOT_DIR, "cfg/exp", exp_file))
            self.params = override_params(self.params, exp_override)

        self.device = rospy.get_param("~device")

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        # Image callback
        self.pub = rospy.Publisher("/random_float", Float32, queue_size=10)

        self.camera_handler = {}
        for cam in self.camera_topics:
            # Initialize camera handler for given cam
            self.camera_handler[cam] = {}
            # Store camera name
            self.camera_topics[cam]["name"] = cam
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
                    queue_size=1,
                    buff_size=2**24,
                )
            else:
                image_sub = rospy.Subscriber(
                    self.camera_topics[cam]["image_topic"],
                    Image,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                    buff_size=2**24,
                )
            self.camera_handler[cam]["image_sub"] = image_sub

            # Set publishers
            input_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/image_input", Image, queue_size=10)
            trav_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/traversability", Image, queue_size=10)
            conf_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/confidence", Image, queue_size=10)
            info_pub = rospy.Publisher(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo, queue_size=10)
            imagefeat_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/camera_info", ImageFeatures, queue_size=10
            )

            self.camera_handler[cam]["input_pub"] = input_pub
            self.camera_handler[cam]["trav_pub"] = trav_pub
            self.camera_handler[cam]["conf_pub"] = conf_pub
            self.camera_handler[cam]["info_pub"] = info_pub
            self.camera_handler[cam]["imagefeat_pub"] = imagefeat_pub

    @torch.no_grad()
    def image_callback(self, image_msg: Image, cam: str):
        """Main callback to process incoming images

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
        """
        # convert image message to torch image
        torch_image = rc.ros_image_to_torch(image_msg, device=self.device)
        image_projector = ImageProjector(
            K=torch.eye(4, device=self.device)[None],
            h=torch_image.shape[1],
            w=torch_image.shape[2],
            new_h=self.network_input_image_height,
            new_w=self.network_input_image_width,
        )
        torch_image = image_projector.resize_image(torch_image)
        C, H, W = torch_image.shape

        edges, feat, seg, center, dense_feat = self.feature_extractor.extract(
            img=torch_image[None], return_centers=False, return_dense_features=True
        )

        # Evaluate traversability
        data = Data(x=dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1]))
        prediction = self.model.forward(data)
        out_trav = prediction.reshape(H, W, -1)[:, :, 0]
        self.pub.publish(1.0)

        # Publish traversability
        msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
        msg.header = image_msg.header
        msg.width = out_trav.shape[0]
        msg.height = out_trav.shape[1]
        self.camera_handler[cam]["trav_pub"].publish(msg)

        msg = ImageFeatures()
        msg.header = image_msg.header
        msg.segmentation = rc.numpy_to_ros_image(seg.cpu().numpy().astype(np.int32), "passthrough")
        msg.segmentation.header = image_msg.header
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

        # # Publish the message
        # TODO take care of -1 in the segments and what is the correct encoding to use
        # self.camera_handler[cam]["imagefeat_pub"].publish(msg)


if __name__ == "__main__":
    node_name = "wvn_feature_extractor_node"
    os.system(
        f"rosparam load {WVN_ROOT_DIR}/wild_visual_navigation_ros/config/wild_visual_navigation/default.yaml {node_name}"
    )
    os.system(
        f"rosparam load {WVN_ROOT_DIR}/wild_visual_navigation_ros/config/wild_visual_navigation/inputs/alphasense_compressed.yaml {node_name}"
    )
    rospy.init_node(node_name)
    wvn = WvnFeatureExtractor()
    rospy.spin()
