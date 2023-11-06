""" 
Main node to process ros messages, publish the relevant topics, train the model...
 """
from BaseWVN.utils import NodeForROS
import ros_converter as rc

from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray,Header
from visualization_msgs.msg import Marker
# from wild_visual_navigation_msgs.msg import StampedFloat32MultiArray
import os
import rospy

import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys


class FeatExtractor(NodeForROS):
    def __init__(self):
        super().__init__()

        # Init Camera handler
        self.camera_handler = {}
        self.system_events = {}

        # Initialize ROS nodes
        self.ros_init()

    
    def ros_init(self):
        """ 
        start ros subscribers and publishers and filter/process topics
        """
        
        print("Start waiting for AnymalState topic being published!")
        rospy.wait_for_message(self.camera_topic, CompressedImage)
        
        # Camera subscriber
        camera_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.camera_callback, queue_size=20)

        # Fill in handler
        self.camera_handler['name'] = self.camera_topic
        self.camera_handler['img_sub'] = camera_sub

        # Results publisher
        input_pub=rospy.Publisher('/vd_pipeline/image_input', Image, queue_size=10)
        fric_pub=rospy.Publisher('/vd_pipeline/friction', Image, queue_size=10)
        stiff_pub=rospy.Publisher('/vd_pipeline/stiffness', Image, queue_size=10)
        conf_pub=rospy.Publisher('/vd_pipeline/confidence', Image, queue_size=10)
        info_pub=rospy.Publisher('/vd_pipeline/camera_info', CameraInfo, queue_size=10)
        freq_pub=rospy.Publisher('/test', Float32, queue_size=10)
        # Fill in handler
        self.camera_handler['input_pub']=input_pub
        self.camera_handler['fric_pub']=fric_pub
        self.camera_handler['stiff_pub']=stiff_pub
        self.camera_handler['conf_pub']=conf_pub
        self.camera_handler['info_pub']=info_pub
        self.camera_handler['freq_pub']=freq_pub
        # TODO: Add the publisher for the two graphs and services (save/load checkpoint) maybe
        pass
    
    
    def camera_callback(self, img_msg:CompressedImage):
        """ 
        callback function for the anymal state subscriber
        """
        self.system_events["state_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        
        try:

            if self.mode == "debug":
                    # pub for testing frequency
                    freq_pub = self.camera_handler['freq_pub']
                    msg=Float32()
                    msg.data=1.0
                    freq_pub.publish(msg)
          
            pass
            print(img_msg.data.shape)
        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)
            self.system_events["state_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass

 
if __name__ == "__main__":
    node_name = "FeatExtractor_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = FeatExtractor()
    rospy.spin()