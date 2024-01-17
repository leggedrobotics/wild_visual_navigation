from BaseWVN.utils import NodeForROS,FeatureExtractor,ConfidenceGenerator,ImageProjector,plot_overlay_image
from BaseWVN.GraphManager import Manager,MainNode,SubNode
import ros_converter as rc
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray,Header
from nav_msgs.msg import Path
from anymal_msgs.msg import AnymalState
from geometry_msgs.msg import PoseStamped, Point,TransformStamped
from visualization_msgs.msg import Marker
from wild_visual_navigation_msgs.msg import SystemState,PhyDecoderOutput,FeatExtractorOutput,FeatInfo
import os
import rospy
import time
import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys
from threading import Thread, Event,Lock
from prettytable import PrettyTable
from termcolor import colored
import PIL.Image
import tf2_ros
from pytictac import ClassTimer, ClassContextTimer, accumulate_time

class FeatExtractor(NodeForROS):
    def __init__(self):
        super().__init__()
        self.step=0
        # Timers to control the rate of the callbacks
        self.last_image_ts = rospy.get_time()
        self.last_supervision_ts = rospy.get_time()
        
        # Init feature extractor
        self.feat_extractor = FeatureExtractor(device=self.device,
                                               segmentation_type=self.segmentation_type,
                                               feature_type=self.feature_type,
                                               input_size=self.input_size,
                                               interp=self.interp,
                                               center_crop=self.center_crop)
           
        # Init Camera handler
        self.camera_handler = {}
        self.system_events = {}
        self.timer=ClassTimer(
            objects=[self,
                     ],
            names=["Main_process"],
            enabled=self.param.general.timestamp
        )
        
        if self.verbose:
            self.log_data = {}
            self.status_thread_stop_event = Event()
            self.status_thread = Thread(target=self.status_thread_loop, name="status")
            self.log_data["Lock"] = Lock()
            self.run_status_thread = True
            self.status_thread.start()

        # Initialize ROS nodes
        self.ros_init()
        
        
        print("Feat_extractor node initialized!")
    
    def shutdown_callback(self, *args, **kwargs):
        self.run_status_thread = False
        self.status_thread_stop_event.set()
        self.status_thread.join()
        if self.param.general.timestamp:
            print(self.timer)
        rospy.signal_shutdown(f"FeatExtractor Node killed {args}")
        sys.exit(0)
    
    def clear_screen(self):
        # For Windows
        if os.name == 'nt':
            _ = os.system('cls')
        # For Linux and Mac
        else:
            _ = os.system('clear')
        
    def status_thread_loop(self):
        # Log loop
        # TODO: make the table prettier, into columns...
        while self.run_status_thread:
            self.status_thread_stop_event.wait(timeout=0.01)
            if self.status_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]
            with self.log_data["Lock"]:  # Acquire the lock
                for k, v in self.log_data.items():
                    if "Lock" in k:
                        continue
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
            # self.clear_screen()
            print(x)
            time.sleep(0.1)
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
            with self.log_data["Lock"]:
                self.log_data[f"time_last_model"] = -1
                self.log_data[f"num_model_updates"] = -1
                self.log_data[f"num_images"] = 0
                self.log_data[f"time_last_image"] = -1
                self.log_data[f"image_callback"] = "N/A"

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
        K_scaled=rc.scale_intrinsic(K,ratio_x,ratio_y)
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
        
        # camera tf broadcaster
        self.camera_br=tf2_ros.TransformBroadcaster()

        # Results publisher
        input_pub=rospy.Publisher('/vd_pipeline/image_input', Image, queue_size=10)
        fric_pub=rospy.Publisher('/vd_pipeline/friction', Image, queue_size=10)
        stiff_pub=rospy.Publisher('/vd_pipeline/stiffness', Image, queue_size=10)
        conf_pub=rospy.Publisher('/vd_pipeline/confidence', Image, queue_size=10)
        info_pub=rospy.Publisher('/vd_pipeline/camera_info', CameraInfo, queue_size=10)
        freq_pub=rospy.Publisher('/test', Float32, queue_size=10)
        feat_ext_pub=rospy.Publisher('/vd_pipeline/feat_ext_out', FeatExtractorOutput, queue_size=10)
        system_state_pub=rospy.Publisher('/vd_pipeline/system_state', SystemState, queue_size=10)
        # Fill in handler
        self.camera_handler['input_pub']=input_pub
        self.camera_handler['fric_pub']=fric_pub
        self.camera_handler['stiff_pub']=stiff_pub
        self.camera_handler['conf_pub']=conf_pub
        self.camera_handler['info_pub']=info_pub
        self.camera_handler['freq_pub']=freq_pub
        self.camera_handler['feat_ext_pub']=feat_ext_pub
        self.camera_handler['system_state_pub']=system_state_pub
        pass
    
    @accumulate_time
    def camera_callback(self, img_msg:CompressedImage,state_msg:AnymalState,cam:str):
        """ 
        callback function for the anymal state subscriber
        """
        self.system_events["camera_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        
        try:
            # Run the callback so as to match the desired rate
            ts = img_msg.header.stamp.to_sec()
            if abs(ts - self.last_image_ts) < 1.0 / self.image_callback_rate:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data[f"image_callback"] = "skipping"
                return
            else:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data[f"image_callback"] = "processing"
            
            self.log_data[f"ros_time_now"] = rospy.get_time()
            if "debug" in self.mode:
                # pub for testing frequency
                freq_pub = self.camera_handler['freq_pub']
                msg=Float32()
                msg.data=1.0
                freq_pub.publish(msg)
            
            # TODO:load MLP and confidence generator params if possible,
            # don't know if needed
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
            _, seg,transformed_img,compressed_feats=self.feat_extractor.extract(img_torch)
            
            feat_msgs=[]
            for (ratioH, ratioW), feature_matrix in compressed_feats.items():
                B,C,H,W=feature_matrix.shape
                feat_msg=FeatInfo()
                feat_msg.ratioH=ratioH
                feat_msg.ratioW=ratioW
                feat_msg.feat=feature_matrix.cpu().numpy().flatten().tolist()
                feat_msg.height=H
                feat_msg.width=W
                feat_msg.channel=C
                feat_msgs.append(feat_msg)

            # # tolist is expensive
           
            msg=FeatExtractorOutput()
            msg.header=img_msg.header
            # msg.features=feat_msg
            msg.pose_base_in_world=pose_base_in_world.flatten().tolist()
            msg.pose_camera_in_world=pose_cam_in_world.flatten().tolist()
            # msg.resized_image=rc.numpy_to_ros_image((transformed_img.squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8), "rgb8")
            msg.segment_type=self.segmentation_type
            if self.segmentation_type!="pixel": 
                msg.segments=seg.cpu().numpy().flatten().tolist()
            msg.resized_K=self.camera_handler["K_scaled"].cpu().numpy().flatten().tolist()
            msg.resized_height=self.camera_handler["H_scaled"]
            msg.resized_width=self.camera_handler["W_scaled"]
            
            self.camera_handler['feat_ext_pub'].publish(msg)
            
            
            # TODO: the log maybe need to change
            if self.verbose:
                with self.log_data["Lock"]:
                    self.log_data[f"num_images"]+=1
                    self.log_data[f"time_last_image"]=rospy.get_time()
                

            self.system_events["image_callback_state"] = {"time": rospy.get_time(), "value": "executed successfully"}
            self.last_image_ts = ts 
        except Exception as e:
            traceback.print_exc()
            print("error camera callback", e)
            self.system_events["camera_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass
    
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

    node_name = "Feat_extractor_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
   
    feat_node = FeatExtractor()

    rospy.spin()