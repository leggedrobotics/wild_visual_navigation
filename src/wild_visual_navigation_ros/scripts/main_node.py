""" 
Main node to process ros messages, publish the relevant topics, train the model...
 """
from BaseWVN import ParamCollection
from Phy_Decoder import initialize_models,prepare_padded_input,RNNInputBuffer

from pytictac import ClassTimer, ClassContextTimer, accumulate_time
import ros_converter as rc
from anymal_msgs.msg import AnymalState
from geometry_msgs.msg import PoseStamped, Point, TwistStamped
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray,Header
from threading import Thread, Event
from visualization_msgs.msg import Marker
# from wild_visual_navigation_msgs.msg import StampedFloat32MultiArray
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
        
        # Read the parameters from the config file
        self.param=ParamCollection()
        self.import_params()
        self.color_palette = sns.color_palette(self.palette,as_cmap=True)
        self.timers=ClassTimer(objects=[self],names=['WvnRosInterface'],enabled=self.print_time)
        
        # Init for PHYSICS DECODERs
        self.step=0
        self.env_num=1
        self.fric_predictor,self.stiff_predictor,self.predictor_cfg=initialize_models()
        self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
        self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
        self.input_buffers = {0: RNNInputBuffer()}

        # Init Camera handler
        self.camera_handler = {}
        self.system_events = {}

        # Initialize ROS nodes
        self.ros_init()

        
    @accumulate_time
    def import_params(self):
        """ 
        read cfg file and import the parameters 
        """
        # TOPIC-REALTED PARAMETERS
        self.anymal_bag_name = self.param.roscfg.anymal_bag_name
        self.anymal_state_topic = self.param.roscfg.anymal_state_topic
        self.feet_list = self.param.roscfg.feet_list
        self.camera_bag_name = self.param.roscfg.camera_bag_name
        self.camera_topic = self.param.roscfg.camera_topic
        self.phy_decoder_input_topic = self.param.roscfg.phy_decoder_input_topic
        self.camera_info_topic = self.param.roscfg.camera_info_topic

        # Frames
        self.fixed_frame = self.param.roscfg.fixed_frame
        self.base_frame = self.param.roscfg.base_frame
        self.footprint_frame = self.param.roscfg.footprint_frame

        # Robot dimensions
        self.robot_length = self.param.roscfg.robot_length
        self.robot_height = self.param.roscfg.robot_height
        self.robot_width = self.param.roscfg.robot_width
        self.robot_max_velocity = self.param.roscfg.robot_max_velocity

        # THREAD PARAMETERS
        self.image_callback_rate = self.param.thread.image_callback_rate
        self.proprio_callback_rate = self.param.thread.proprio_callback_rate
        self.learning_thread_rate = self.param.thread.learning_rate
        self.logging_thread_rate = self.param.thread.logging_rate

        # RUN PARAMETERS
        self.device = self.param.run.device
        self.mode = self.param.run.mode
        self.palette = self.param.run.palette
        self.print_time = self.param.run.print_time
    
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
        # Timers to control the rate of the publishers
        self.last_image_ts = rospy.get_time()
        self.last_proprio_ts = rospy.get_time()
        
        
        
        self.state_sub.registerCallback(self.state_callback)

        # Camera subscriber
        camera_sub = message_filters.Subscriber(self.camera_topic, CompressedImage)
        camera_info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
        self.camera_sub = message_filters.ApproximateTimeSynchronizer([camera_sub, camera_info_sub], queue_size=10,slop=0.1)
        # self.camera_sub.registerCallback(self.camera_callback)

        # Fill in handler
        self.camera_handler['name'] = self.camera_topic
        self.camera_handler['img_sub'] = camera_sub
        self.camera_handler['info_sub'] = camera_info_sub
        self.camera_handler['img_synced']=self.camera_sub

        # Results publisher
        input_pub=rospy.Publisher('/vd_pipeline/image_input', Image, queue_size=10)
        fric_pub=rospy.Publisher('/vd_pipeline/friction', Image, queue_size=10)
        stiff_pub=rospy.Publisher('/vd_pipeline/stiffness', Image, queue_size=10)
        conf_pub=rospy.Publisher('/vd_pipeline/confidence', Image, queue_size=10)
        info_pub=rospy.Publisher('/vd_pipeline/camera_info', CameraInfo, queue_size=10)
        freq_pub=rospy.Publisher('/test', Float32, queue_size=10)
        phy_val_pub=rospy.Publisher('/vd_pipeline/phy_val_pred', Float32MultiArray, queue_size=10)
        # stamped_debug_info_pub=rospy.Publisher('/stamped_debug_info', StampedFloat32MultiArray, queue_size=10)
        # Fill in handler
        self.camera_handler['input_pub']=input_pub
        self.camera_handler['fric_pub']=fric_pub
        self.camera_handler['stiff_pub']=stiff_pub
        self.camera_handler['conf_pub']=conf_pub
        self.camera_handler['info_pub']=info_pub
        # self.camera_handler['stamped_debug_info_pub']=stamped_debug_info_pub
        self.camera_handler['freq_pub']=freq_pub
        self.camera_handler['phy_val_pub']=phy_val_pub
        # TODO: Add the publisher for the two graphs and services (save/load checkpoint) maybe
        pass
    
    @accumulate_time
    def query_tf(self, parent_frame: str, child_frame: str, stamp: Optional[rospy.Time] = None, from_message: Optional[AnymalState] = None):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
            from_message (AnymalState, optional): AnymalState message containing the TFs
        """
        if from_message is None:
            # Original behavior: look up the TF from the TF tree
            if stamp is None:
                stamp = rospy.Time(0)
            try:
                res = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp)
                trans = (res.transform.translation.x, res.transform.translation.y, res.transform.translation.z)
                rot = np.array(
                    [
                        res.transform.rotation.x,
                        res.transform.rotation.y,
                        res.transform.rotation.z,
                        res.transform.rotation.w,
                    ]
                )
                rot /= np.linalg.norm(rot)
                return (trans, tuple(rot))
            except Exception as e:
                print("Error in query tf: ", e)
                rospy.logwarn(f"Couldn't get tf between {parent_frame} and {child_frame}")
                return (None, None)
        else:
            # New behavior: extract the TF from the AnymalState message
            if child_frame not in self.feet_list:
                for frame_transform in from_message.frame_transforms:
                    if frame_transform.header.frame_id == parent_frame and frame_transform.child_frame_id == child_frame:
                        trans = (frame_transform.transform.translation.x,
                                frame_transform.transform.translation.y,
                                frame_transform.transform.translation.z)
                        rot = np.array(
                            [
                                frame_transform.transform.rotation.x,
                                frame_transform.transform.rotation.y,
                                frame_transform.transform.rotation.z,
                                frame_transform.transform.rotation.w,
                            ]
                        )
                        rot /= np.linalg.norm(rot)
                        return (trans, tuple(rot))                   
            else:
                for foot_transform in from_message.contacts:
                    if foot_transform.name == child_frame and foot_transform.header.frame_id==parent_frame:
                        trans=(foot_transform.position.x,
                               foot_transform.position.y,
                               foot_transform.position.z)
                        # FOOT is merely a point, no rotation
                        rot=np.array([0,0,0,1])
                        return (trans,tuple(rot))

            # If the requested TF is not found in the message
            rospy.logwarn(f"Couldn't find tf between {parent_frame} and {child_frame} in the provided message")
            return (None, None)

        
    @accumulate_time
    def state_callback(self, anymal_state_msg:AnymalState, phy_decoder_input_msg:Float32MultiArray):
        """ 
        callback function for the anymal state subscriber
        """
        self.step+=1
        self.system_events["state_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        
        try:

            if self.mode == "debug":
                    # pub for testing frequency
                    freq_pub = self.camera_handler['freq_pub']
                    msg=Float32()
                    msg.data=1.0
                    freq_pub.publish(msg)
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
                self.camera_handler['phy_val_pub'].publish(Float32MultiArray(data=new_priv))


            pass
        
        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)
            self.system_events["state_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass

 
if __name__ == "__main__":
    node_name = "BaseWVN"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node("BaseWVN")
    wvn = WvnRosInterface()
    rospy.spin()
