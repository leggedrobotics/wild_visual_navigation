""" 
Main node to process ros messages, publish the relevant topics, train the model...
 """
from BaseWVN import ParamCollection

from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray
from threading import Thread, Event
from visualization_msgs.msg import Marker
import message_filters
import os
import rospy
import seaborn as sns
import tf
import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys
import tf2_ros


if torch.cuda.is_available():
    torch.cuda.empty_cache()

class WvnRosInterface:
    def __init__(self):
        # Timers to control the rate of the publishers
        self.last_image_ts = rospy.get_time()
        self.last_proprio_ts = rospy.get_time()
        self.param=ParamCollection()

    
    def import_params(self):
        """ 
        read cfg file and import the parameters 
        """
        self.anymal_state_topic = self.param.roscfg.anymal_state_topic
        self.camera_topic = self.param.roscfg.camera_topic
        self.phy_decoder_input_topic = self.param.roscfg.phy_decoder_input_topic
