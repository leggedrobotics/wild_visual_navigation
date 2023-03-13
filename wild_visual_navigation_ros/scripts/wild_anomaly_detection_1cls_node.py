#!/usr/bin/env python3
"""
Wild anomaly detection inference node, method 1.

Author: Robin Schmid
Date: Feb 2023
"""
import os
import cv2
import copy

import numpy as np
import torch
import torchvision

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from wild_visual_navigation.learning.utils.load_models import load_model_1cls, create_extractor, Timer
from wild_visual_navigation import WVN_ROOT_DIR

# WAD_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
WAD_ROOT_DIR = WVN_ROOT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AnomalyInference(object):
    def __init__(self):
        rospy.init_node("anomaly_inference")

        # Load params
        self.overlay_alpha = rospy.get_param("~overlay_alpha", default=0.35)
        self.threshold = rospy.get_param("~threshold", default=0.25)
        self.unknown_interval = rospy.get_param("~unknown_interval", default=[-0.02, 0.02])
        self.running_median_prior = rospy.get_param("~running_median_prior", default=550)
        self.running_std_prior = rospy.get_param("~running_std_prior", default=400)
        self.img_size = rospy.get_param("~img_size", default=448)
        self.color_safe = rospy.get_param("~color_safe", default=[0, 1, 0])
        self.color_unsafe = rospy.get_param("~color_unsafe", default=[1, 0, 0])
        self.color_unknown = rospy.get_param("~color_unknown", default=[1.0, 0.6, 0.0])
        self.visu = rospy.get_param("~visu", default=True)
        self.debug = rospy.get_param("~debug", default=True)
        self.mask_img = rospy.get_param("~mask_img", default=False)
        self.rescale_orig = rospy.get_param("~rescale_orig", default=False)
        self.img_input_topic = rospy.get_param("~img_input_topic", default=None)
        self.only_img = rospy.get_param("~only_img", default=False)
        self.info_input_topic = rospy.get_param("~info_input_topic", default=None)
        self.slic_num_components = rospy.get_param("~slic_num_components", default=2000)
        self.info_output_topic = rospy.get_param("~info_output_topic", default="/wide_angle_camera_front/camera_info")
        self.bin_img_output_topic = rospy.get_param("~bin_img_output_topic", default="/anonomaly_inference/bin_img")
        self.vis_img_output_topic = rospy.get_param("~vis_img_output_topic", default="/anonomaly_inference/vis_img")
        self.weights_path = os.path.join(WAD_ROOT_DIR, "assets", "weights", "1cls", rospy.get_param("~weights", default=""))
        self._cv_bridge = CvBridge()

        self.new_info_msg = None

        # Load model
        self.model = load_model_1cls(model_path=self.weights_path, device=DEVICE)
        self.feature_extractor = create_extractor(device=DEVICE, input_size=448,
                                                  slic_num_components=self.slic_num_components)

        # Init subscribers and publishers
        if self.only_img:
            self.image_sub = rospy.Subscriber(self.img_input_topic, Image, self.process, queue_size=1,
                                              buff_size=2 ** 24)
            self.info_sub = rospy.Subscriber(self.info_input_topic, CameraInfo, self.info_cb, queue_size=1)
        else:
            self.image_sub = message_filters.Subscriber(self.img_input_topic, Image, queue_size=1, buff_size=2 ** 24)
            self.info_sub = message_filters.Subscriber(self.info_input_topic, CameraInfo, queue_size=1)
            sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub], queue_size=2, slop=0.5)
            sync.registerCallback(self.process)

        self.image_pub_bin = rospy.Publisher(self.bin_img_output_topic, Image, queue_size=1)
        self.image_pub_vis = rospy.Publisher(self.vis_img_output_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(self.info_output_topic, CameraInfo, queue_size=1)

        self.it = 0
        self.count = 0
        self.sum_time = 0

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        rospy.loginfo("Shutdown anomaly inference node")
        return

    def info_cb(self, msg):
        self.new_info_msg = msg

    def convert_ros_msg_to_cv2(self, ros_data, image_encoding='bgr8'):
        self.height = ros_data.height
        self.width = ros_data.width

        try:
            return self._cv_bridge.imgmsg_to_cv2(ros_data, image_encoding)
        except CvBridgeError as e:
            raise e

    def convert_ros_msg_to_tensor(self, msg):
        # Convert ROS msg to cv2 image
        img = self.convert_ros_msg_to_cv2(msg)

        # Convert from uint8 to float32
        img = img.astype(np.float32) / 255.0

        # Convert to torch tensor
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.to(DEVICE)

        # Resize image to square shape
        if self.rescale_orig:
            img = torchvision.transforms.Resize((self.img_size, self.img_size),
                                                torchvision.transforms.InterpolationMode.BILINEAR)(img)
        else:
            img = torchvision.transforms.CenterCrop(self.img_size)(img)

        return img

    def process(self, img_msg):
        last_image_time = rospy.Time.now().to_sec()

        # Convert ROS msg to tensor
        img = self.convert_ros_msg_to_tensor(img_msg)

        # Extract features
        feat, seg = self.feature_extractor.extract(img[None], return_centers=False)

        # Inference
        bin_img = self.inference(feat, seg)

        if self.visu:
            img = img.permute(1, 2, 0).cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            seg = np.array(bin_img.cpu().numpy(), dtype=np.float32)

            overlay = img.copy()
            overlay[:] = self.color_safe - np.multiply.outer(
                seg * (1 / self.threshold),
                self.color_safe,
            )
            overlay[seg >= self.threshold] = 0

            # Unsafe regions
            seg_new = copy.deepcopy(seg)
            seg_new[seg < self.threshold] = 0
            overlay[:] += np.multiply.outer(
                seg_new * (1 / (1 - self.threshold)),
                self.color_unsafe,
            )

            # Unknown regions
            overlay[np.where((seg > self.threshold + self.unknown_interval[0]) &
                             (seg < self.threshold + self.unknown_interval[1]), 0, 1) == 0] = self.color_unknown

            # Overlay safe and unsafe regions with image
            res = cv2.addWeighted(overlay, self.overlay_alpha, img, 1 - self.overlay_alpha, 0.0)
            res = cv2.convertScaleAbs(res, alpha=(255.0))

            # Convert to opencv image
            vis_img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

            if self.rescale_orig:
                vis_img = torch.from_numpy(vis_img).permute(2, 0, 1).unsqueeze(0)
                vis_img = torchvision.transforms.Resize((self.height, self.width),
                                                        torchvision.transforms.InterpolationMode.BILINEAR)(vis_img)
                vis_img = vis_img.squeeze().permute(1, 2, 0).numpy()

            self.image_pub_vis.publish(self._cv_bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))

        if self.rescale_orig:
            bin_img = bin_img.unsqueeze(0)
            bin_img = torchvision.transforms.Resize((self.height, self.width),
                                                    torchvision.transforms.InterpolationMode.NEAREST)(bin_img)

        # Publish images
        if bin_img is not None:
            out_img = torch.where(bin_img.squeeze() >= self.threshold, 0.0, 1.0)
            out_img = self._cv_bridge.cv2_to_imgmsg(out_img.cpu().numpy(), encoding="passthrough")

            # Set correct header
            out_img.header = img_msg.header
            self.image_pub_bin.publish(out_img)

            if self.new_info_msg is not None:
                # Update the info_msg header time stamp
                self.new_info_msg.header.stamp = out_img.header.stamp
                self.info_pub.publish(self.new_info_msg)

        if self.debug:
            self.count += 1
            if self.count > 1:
                self.it += 1
                self.sum_time += rospy.Time.now().to_sec() - last_image_time
                mean_time = self.sum_time / self.it
                rospy.logwarn(f"Mean Inference Rate: {1 / mean_time:.3f} Hz")

    def inference(self, feat, seg):
        with torch.no_grad():
            # Compute loss for feature
            u, log_det = self.model.forward(feat)
            prior_logprob = self.model.logprob(u)

            losses = -(prior_logprob.sum(1) + log_det)

            # Clip in range
            losses = torch.clip(losses, self.running_median_prior - 2 * self.running_std_prior,
                                self.running_median_prior + 2 * self.running_std_prior)

            # Normalize between 0 and 1
            losses = (losses - torch.min(losses)) / (torch.max(losses) - torch.min(losses))

            # Deal with 0 value predictions
            losses[losses == 0] = 1e-6

            # Assign loss to each pixel
            img = losses[seg.reshape(-1)]

            img = img.reshape((self.img_size, self.img_size))

            return img


if __name__ == "__main__":
    rospy.loginfo("\n\n\nStarting anomaly inference node...")
    rospy.logwarn("\nUsing 1 class mode")
    rospy.loginfo(f"\nUsing device: {DEVICE}\n")

    inference = AnomalyInference()
    rospy.spin()
