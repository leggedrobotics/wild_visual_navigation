#!/env/bin/python3
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#

from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageConverter:
    def __init__(self, cam):
        self.image_pub = rospy.Publisher(f"/alphasense_driver_ros/{cam}_corrected", Image, queue_size=10)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(f"/alphasense_driver_ros/{cam}", Image, self.callback)

    def callback(self, msg):
        msg.encoding = "bayer_gbrg8"
        self.image_pub.publish(msg)


def main(args):
    rospy.init_node("image_converter", anonymous=True)
    cam = rospy.get_param("~cam", "cam4")
    ic = ImageConverter(cam)  # noqa: F841
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
