#!/usr/bin/env python

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image as PILImage


class ImageSaver:
    def __init__(self):
        rospy.init_node("image_saver", anonymous=True)
        self.bridge = CvBridge()
        self.count = 0

        self.dir_path = "/Data/open_source_dataset"
        # Create directories to store images
        # Subscriber for the images with approximate time synchronization
        self.trav_sub = Subscriber("/wild_visual_navigation_visu_traversability_rear/traversability_overlayed", Image)
        self.raw_sub = Subscriber("/wide_angle_camera_rear/image_color_rect_resize", Image)
        self.sync = ApproximateTimeSynchronizer([self.trav_sub, self.raw_sub], queue_size=1, slop=0.3)
        self.sync.registerCallback(self.callback)

    def callback(self, trav_msg, raw_msg):
        try:
            trav_cv2 = self.bridge.imgmsg_to_cv2(trav_msg, desired_encoding="passthrough")
            raw_cv2 = self.bridge.imgmsg_to_cv2(raw_msg, desired_encoding="passthrough")

            # Convert images to RGBA format
            trav_rgba = cv2.cvtColor(trav_cv2, cv2.COLOR_RGB2BGR)
            raw_rgba = cv2.cvtColor(raw_cv2, cv2.COLOR_BGR2BGRA)

            # Save traversability image
            trav_filename = os.path.join(self.dir_path, f"{self.count}_trav.png")
            cv2.imwrite(trav_filename, trav_rgba)
            rospy.loginfo(f"Saved traversability image: {trav_filename}")

            # Save raw image
            raw_filename = os.path.join(self.dir_path, f"{self.count}_raw.png")
            cv2.imwrite(raw_filename, raw_rgba)
            rospy.loginfo(f"Saved raw image: {raw_filename}")

            self.count += 1
        except Exception as e:
            rospy.logerr(f"Error processing images: {e}")


if __name__ == "__main__":
    try:
        ImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
