""" 
Main node to process ros messages, publish the relevant topics, train the model...
 """
from BaseWVN.utils import NodeForROS,FeatureExtractor,ConfidenceGenerator
import ros_converter as rc
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray,Header
from anymal_msgs.msg import AnymalState
from visualization_msgs.msg import Marker
from wild_visual_navigation_msgs.msg import FeatExtractorOutput
import os
import rospy

import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys
from threading import Thread, Event
from prettytable import PrettyTable
from termcolor import colored

class MainProcess(NodeForROS):
    def __init__(self):
        super().__init__()
        self.step=0

        # Init feature extractor
        self.feat_extractor = FeatureExtractor(device=self.device,
                                               segmentation_type=self.segmentation_type,
                                               feature_type=self.feature_type,
                                               input_size=self.input_size,
                                               interp=self.interp,
                                               center_crop=self.center_crop)
        # Init confidence generator
        self.confidence_generator = ConfidenceGenerator(std_factor=self.confidence_std_factor,
                                                        method=self.method,
                                                        log_enabled=self.log_enabled,
                                                        log_folder=self.log_folder,
                                                        device=self.device)

        # TODO:Load the visual decoder (to device, eval mode)
        self.visual_decoder = []

        # Init Camera handler
        self.camera_handler = {}
        self.system_events = {}

        if self.verbose:
            self.log_data = {}
            # self.status_thread_stop_event = Event()
            # self.status_thread = Thread(target=self.status_thread_loop, name="status")
            # self.run_status_thread = True
            # self.status_thread.start()

        # Initialize ROS nodes
        self.ros_init()
    
    def shutdown_callback(self, *args, **kwargs):
        self.run_status_thread = False
        self.status_thread_stop_event.set()
        self.status_thread.join()

        rospy.signal_shutdown(f"FeatExtractor Node killed {args}")
        sys.exit(0)
    
    def status_thread_loop(self):
        # Learning loop
        while self.run_status_thread:
            self.status_thread_stop_event.wait(timeout=0.01)
            if self.status_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]

            for k, v in self.log_data.items():
                if "time" in k:
                    d = t - v
                    if d < 0:
                        c = "red"
                    if d < 0.2:
                        c = "green"
                    elif d < 1.0:
                        c = "yellow"
                    else:
                        c = "red"
                    x.add_row([k, colored(round(d, 2), c)])
                else:
                    x.add_row([k, v])
            print(x)
            # try:
            #    rate.sleep()
            # except Exception as e:
            #    rate = rospy.Rate(self.ros_params.status_thread_rate)
            #    print("Ignored jump pack in time!")
        self.status_thread_stop_event.clear()
    
    def ros_init(self):
        """ 
        start ros subscribers and publishers and filter/process topics
        """
        
        if self.verbose:
            # DEBUG Logging
            self.log_data[f"time_last_model"] = -1
            self.log_data[f"num_model_updates"] = -1
            cam=self.camera_topic
            self.log_data[f"num_images_{cam}"] = 0
            self.log_data[f"time_last_image_{cam}"] = -1

        print("Start waiting for Camera topic being published!")
        # Camera info
        camera_info_msg = rospy.wait_for_message(
            self.camera_info_topic, CameraInfo)
        K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self.device)
        
        self.camera_handler["camera_info"] = camera_info_msg
        self.camera_handler["K"] = K
        self.camera_handler["H"] = H
        self.camera_handler["W"] = W
        self.camera_handler["distortion_model"] = camera_info_msg.distortion_model

        # update size info in the feature extractor
        self.feat_extractor.set_original_size(original_height=H, original_width=W)
        ratio_x,ratio_y=self.feat_extractor.resize_ratio

        # scale the intrinsic matrix
        K_scaled=self.scale_intrinsic(K,ratio_x,ratio_y)
        W_scaled,H_scaled=self.feat_extractor.new_size
        # update the camera info
        self.camera_handler["K_scaled"] = K_scaled
        self.camera_handler["H_scaled"] = H_scaled
        self.camera_handler["W_scaled"] = W_scaled

        # Camera and anymal state subscriber
        camera_sub=message_filters.Subscriber(self.camera_topic, CompressedImage)
        anymal_state_sub = message_filters.Subscriber(self.anymal_state_topic, AnymalState)
        sync= message_filters.ApproximateTimeSynchronizer([camera_sub,anymal_state_sub],queue_size=200,slop=0.2)
        sync.registerCallback(self.camera_callback,self.camera_topic)

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
    
    
    def camera_callback(self, img_msg:CompressedImage,state_msg:AnymalState,cam:str):
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
            
            # load MLP and confidence generator params if possible
            self.load_model()
            
            # prepare tf from base to camera
            transform=state_msg.pose
            trans=(transform.pose.position.x,
                   transform.pose.position.y,
                   transform.pose.position.z)
            rot=(transform.pose.orientation.x,
                 transform.pose.orientation.y,
                 transform.pose.orientation.z,
                 transform.pose.orientation.w)
            suc, pose_base_in_world = rc.ros_tf_to_numpy((trans,rot))
            if not suc:
                self.system_events["camera_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_base_in_world",
                }
                return
            
            # transform the camera pose from base to world
            pose_cam_in_base=self.param.roscfg.rear_camera_in_base
            pose_cam_in_world=np.matmul(pose_base_in_world,pose_cam_in_base)
            self.camera_handler["pose_cam_in_world"]=pose_cam_in_world

            # prepare image
            img_torch = rc.ros_image_to_torch(img_msg, device=self.device)
            img_torch = img_torch[None]
            features, seg,transformed_img=self.feat_extractor.extract(img_torch)
            
            # tolist is expensive
            # msg=FeatExtractorOutput()
            # msg.header=img_msg.header
            # msg.features=features.reshape(-1).cpu().numpy()
            # msg.segments=seg.cpu().numpy().flatten().tolist()
            # msg.resized_image=transformed_img.cpu().numpy().flatten().tolist()
            # msg.ori_camera_info=self.camera_handler["camera_info"]
            # msg.resized_K=self.camera_handler["K_scaled"].cpu().numpy().flatten().tolist()
            # msg.resized_height=self.camera_handler["H_scaled"]
            # msg.resized_width=self.camera_handler["W_scaled"]
            
            

            if self.verbose:
                self.log_data[f"num_images_{cam}"]+=1
                self.log_data[f"time_last_image_{cam}"]=rospy.get_time()
        except Exception as e:
            traceback.print_exc()
            print("error camera callback", e)
            self.system_events["camera_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass

    def scale_intrinsic(self,K:torch.tensor,ratio_x,ratio_y):
        """ 
        scale the intrinsic matrix
        """
        # dimension check of K
        if K.shape[2]!=3 or K.shape[1]!=3:
            raise ValueError("The dimension of the intrinsic matrix is not 4x4!")
        K_scaled = K.clone()
        K_scaled[:,0,0]=K[:,0,0]*ratio_x
        K_scaled[:,0,2]=K[:,0,2]*ratio_x
        K_scaled[:,1,1]=K[:,1,1]*ratio_y
        K_scaled[:,1,2]=K[:,1,2]*ratio_y
        return K_scaled
    
    def load_model(self):
        """ 
        load the model from the checkpoint
        """
        try:
            self.step+=1
            if self.step%100==0:
                print(f"Loading model from checkpoint {self.step}")
                pass
        except Exception as e:
            if self.verbose:
                print(f"Model Loading Failed: {e}")


if __name__ == "__main__":

    node_name = "Main_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = MainProcess()
    rospy.spin()