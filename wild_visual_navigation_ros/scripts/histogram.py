import rospy
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np


class HistogramNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, self.callback)

    def callback(self, msg):
        # Convert the ROS Image message to a numpy array
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Compute the histogram of the image
        hist = np.histogram(img.flatten(), bins=100)

        # Plot the histogram using matplotlib
        plt.hist(img, bins=100, range=[0, 1])
        plt.savefig("histogram.png")


if __name__ == "__main__":
    rospy.init_node("histogram_node")
    node = HistogramNode()
    rospy.spin()
