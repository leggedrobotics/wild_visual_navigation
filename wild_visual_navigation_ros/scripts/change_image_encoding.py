#!/env/bin/python3

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
    def __init__(self):
        rospy.init_node("image_converter", anonymous=True)
        self.image_pub = rospy.Publisher("/alphasense_driver_ros/cam4/corrected_encoding", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/alphasense_driver_ros/cam4", Image, self.callback)

    def callback(self, msg):
        msg.encoding = "bayer_gbrg8"
        self.image_pub.publish(msg)


def main(args):

    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
