#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import message_filters
import rospy
from sensor_msgs.msg import Image  # , CameraInfo  # , CompressedImage
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation.visu import LearningVisualizer
import sys


class ImageOverlayNode:
    def __init__(self):
        self.image_sub_topic = rospy.get_param("~image_sub_topic")
        self.value_sub_topic = rospy.get_param("~value_sub_topic")
        self.image_pub_topic = rospy.get_param("~image_pub_topic")

        self._pub = rospy.Publisher(f"~{self.image_pub_topic}", Image, queue_size=1)
        self._visualizer = LearningVisualizer()

        image_sub = message_filters.Subscriber(self.image_sub_topic, Image)
        trav_sub = message_filters.Subscriber(self.value_sub_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer([image_sub, trav_sub], queue_size=2, slop=0.5)
        sync.registerCallback(self.callback)

    def callback(self, image_msg, trav_msgs):
        torch_image = rc.ros_image_to_torch(image_msg, device="cpu")
        torch_trav = rc.ros_image_to_torch(trav_msgs, device="cpu", desired_encoding="passthrough")
        img_out = self._visualizer.plot_detectron_classification(torch_image, torch_trav.clip(0, 1))
        ros_msg = rc.numpy_to_ros_image(img_out)
        ros_msg.header.stamp = image_msg.header.stamp
        self._pub.publish(ros_msg)


if __name__ == "__main__":
    print("Stated ImageOverlayNode")
    try:
        args = rospy.myargv(argv=sys.argv)
        if "--nr" in args:
            nr_index = args.index("--nr")
            nr = args[nr_index + 1]
        else:
            nr = "0"  # Handle case when no arg is set
        rospy.init_node(f"wild_visual_navigation_visu_{nr}")
    except Exception:
        rospy.init_node("wild_visual_navigation_visu")

    wvn = ImageOverlayNode()
    rospy.spin()
