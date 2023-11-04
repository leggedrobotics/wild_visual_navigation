""" 
Main node to process ros messages, publish the relevant topics, train the model...
 """
from BaseWVN import ParamCollection
from BaseWVN.utils import NodeForROS
from Phy_Decoder import initialize_models,prepare_padded_input,RNNInputBuffer

from pytictac import ClassTimer, ClassContextTimer, accumulate_time
import ros_converter as rc
from anymal_msgs.msg import AnymalState
from geometry_msgs.msg import Pose, Point, Quaternion
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray,Header
from threading import Thread, Event
from visualization_msgs.msg import Marker
from wild_visual_navigation_msgs.msg import PhyDecoderOutput
import message_filters
import os
import rospy
import seaborn as sns
import tf.transformations as tft
import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys


class PhyDecoder(NodeForROS):
    def __init__(self):
        super().__init__()
        
        # Init for PHYSICS DECODERs
        self.step=0
        self.env_num=1
        self.fric_predictor,self.stiff_predictor,self.predictor_cfg=initialize_models()
        self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
        self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
        self.input_buffers = {0: RNNInputBuffer()}

        # Init Camera handler
        self.decoder_handler = {}
        self.system_events = {}

        # Initialize ROS nodes
        self.ros_init()

    def ros_init(self):
        """ 
        start ros subscribers and publishers and filter/process topics
        """
        
        # Anymal state subscriber
        print("Start waiting for AnymalState topic being published!")
        rospy.wait_for_message(self.anymal_state_topic, AnymalState)
        anymal_state_sub = message_filters.Subscriber(self.anymal_state_topic, AnymalState)
        cache_anymal_state = message_filters.Cache(anymal_state_sub, 10,allow_headerless=True)
        
        print("Start waiting for Phy_decoder_input topic being published!")
        rospy.wait_for_message(self.phy_decoder_input_topic, Float32MultiArray)
        phy_decoder_input_sub = message_filters.Subscriber(self.phy_decoder_input_topic, Float32MultiArray)  
        cache_phy_decoder_input = message_filters.Cache(phy_decoder_input_sub, 200,allow_headerless=True)

        self.state_sub=message_filters.ApproximateTimeSynchronizer([anymal_state_sub, phy_decoder_input_sub], queue_size=100,slop=0.1,allow_headerless=True)
        
        print("Current ros time is: ",rospy.get_time())
        
        self.state_sub.registerCallback(self.state_callback)


        # Results publisher
        phy_decoder_pub=rospy.Publisher('/vd_pipeline/phy_decoder_out', PhyDecoderOutput, queue_size=10)
        # stamped_debug_info_pub=rospy.Publisher('/stamped_debug_info', StampedFloat32MultiArray, queue_size=10)
        # Fill in handler
        self.decoder_handler['phy_decoder_pub']=phy_decoder_pub

    
        
    @accumulate_time
    def state_callback(self, anymal_state_msg:AnymalState, phy_decoder_input_msg:Float32MultiArray):
        """ 
        callback function for the anymal state subscriber
        """
        self.step+=1
        self.system_events["state_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        msg=PhyDecoderOutput()
        msg.header=anymal_state_msg.header
        try:

            # Query footprint transform from AnymalState message
            suc, pose_footprint_in_world = rc.ros_tf_to_torch(
                self.query_tf(self.fixed_frame, self.footprint_frame, from_message=anymal_state_msg), device=self.device
            )
            if not suc:
                self.system_events["state_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_footprint_in_base",
                }
                return
            
            # Query 4 feet transforms from AnymalState message
            pose_feet_in_world = {}
            foot_poses=[]
            for foot in self.feet_list:
                suc, pose_foot_in_world = rc.ros_tf_to_torch(
                    self.query_tf(self.fixed_frame, foot, from_message=anymal_state_msg), device=self.device
                )
                if not suc:
                    self.system_events["state_callback_cancled"] = {
                        "time": rospy.get_time(),
                        "value": f"cancled due to pose_{foot}_in_base",
                    }
                    return
                pose_feet_in_world[foot] = pose_foot_in_world
                foot_pose=self.matrix_to_pose(pose_foot_in_world.cpu().numpy())
                foot_poses.append(foot_pose)
            msg.feet_poses=foot_poses

            """ 
            Fric/Stiff-Decoder input topics & prediction
            """
            phy_decoder_input = torch.tensor(phy_decoder_input_msg.data, device=self.device).unsqueeze(0)
            obs, hidden = torch.split(phy_decoder_input, [341, 100], dim=1)
            input_data=obs[:,:341]
            padded_inputs = prepare_padded_input(input_data, self.input_buffers, self.step, self.env_num)    
            padded_input = torch.stack(padded_inputs, dim=0)
            if self.predictor_cfg['reset_hidden_each_epoch']:
                self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
                self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
            with torch.no_grad():
                # Predict using the friction predictor
                fric_pred, self.fric_hidden = self.fric_predictor.get_unnormalized_recon(padded_input, self.fric_hidden)           
                # Predict using the stiffness predictor
                stiff_pred, self.stiff_hidden = self.stiff_predictor.get_unnormalized_recon(padded_input, self.stiff_hidden)
            self.input_buffers[0].add(input_data[0].unsqueeze(0))
            if self.mode == "debug":
            # pub fric and stiff together
                new_priv=torch.cat([fric_pred,stiff_pred],dim=-1)
                new_priv=new_priv[:,-1,:].squeeze(0).cpu().numpy()
                msg.prediction=new_priv
                # self.decoder_handler['phy_decoder_pub'].publish(Float32MultiArray(data=new_priv))


            pass
        
        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)
            self.system_events["state_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass

    @accumulate_time
    def matrix_to_pose(self,matrix):
        # Extract the translation (last column of the matrix)
        position = Point(*matrix[:3, 3])

        # Extract the rotation (upper-left 3x3 submatrix) and convert to quaternion
        quaternion = Quaternion(*tft.quaternion_from_matrix(matrix))

        # Create a Pose message and assign the position and orientation
        pose_msg = Pose()
        pose_msg.position = position
        pose_msg.orientation = quaternion

        return pose_msg

 
if __name__ == "__main__":
    node_name = "Phy_decoder_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = PhyDecoder()
    rospy.spin()
