#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class ImageRotator:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/hdr_cam/image", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/hdr_cam/rotated/image", Image, queue_size=1)

    def image_callback(self, data):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: {0}".format(e))
            return

        # Rotate the image by 90 degrees
        rotated_image = cv2.rotate(cv_image, cv2.ROTATE_180)

        try:
            # Convert OpenCV image to ROS image message
            out_msg = self.bridge.cv2_to_imgmsg(rotated_image, "passthrough")
            out_msg.header = data.header
            self.image_pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: {0}".format(e))


if __name__ == "__main__":
    rospy.init_node("image_rotator_node")
    print("Started node")
    ir = ImageRotator()
    rospy.spin()
