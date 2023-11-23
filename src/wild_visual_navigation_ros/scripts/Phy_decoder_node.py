""" 
Main node to process ros messages, publish the relevant topics, train the model...
 """
from BaseWVN.utils import *
from Phy_Decoder import initialize_models,prepare_padded_input,RNNInputBuffer

import ros_converter as rc
from anymal_msgs.msg import AnymalState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray,Header
from visualization_msgs.msg import Marker,MarkerArray
from wild_visual_navigation_msgs.msg import PhyDecoderOutput,PlaneEdge
import message_filters
import os
import rospy
import tf.transformations as tft
import torch
import numpy as np
from typing import Optional
import traceback



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

        # Init for storing last footprint pose
        self.last_footprint_pose = None
        self.current_footprint_pose = None

        # Init Decoder handler
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
        
        print("Start waiting for Phy_decoder_input topic being published!")
        rospy.wait_for_message(self.phy_decoder_input_topic, Float32MultiArray)
        phy_decoder_input_sub = message_filters.Subscriber(self.phy_decoder_input_topic, Float32MultiArray) 

        self.state_sub=message_filters.ApproximateTimeSynchronizer([anymal_state_sub, phy_decoder_input_sub], queue_size=100,slop=0.1,allow_headerless=True)
        
        print("Current ros time is: ",rospy.get_time())
        
        self.state_sub.registerCallback(self.state_callback)


        # Results publisher
        phy_decoder_pub=rospy.Publisher('/vd_pipeline/phy_decoder_out', PhyDecoderOutput, queue_size=10)
        marker_array_pub = rospy.Publisher('/vd_pipeline/visualization_planes', MarkerArray, queue_size=10)
        # stamped_debug_info_pub=rospy.Publisher('/stamped_debug_info', StampedFloat32MultiArray, queue_size=10)
        # Fill in handler
        self.decoder_handler['phy_decoder_pub']=phy_decoder_pub
        self.decoder_handler['marker_planes_pub']=marker_array_pub
    
        
    def state_callback(self, anymal_state_msg:AnymalState, phy_decoder_input_msg:Float32MultiArray):
        """ 
        callback function for the anymal state subscriber
        """
        
        self.step+=1
        self.system_events["state_callback_received"] = {"time": rospy.get_time(), "value": "message received"}
        msg=PhyDecoderOutput()
        msg.header=anymal_state_msg.header
        # print((rospy.Time.now()-anymal_state_msg.header.stamp)*1e-9)
        try:
            
            transform=anymal_state_msg.pose
            trans=(transform.pose.position.x,
                   transform.pose.position.y,
                   transform.pose.position.z)
            rot=(transform.pose.orientation.x,
                 transform.pose.orientation.y,
                 transform.pose.orientation.z,
                 transform.pose.orientation.w)
            suc, pose_base_in_world = rc.ros_tf_to_numpy((trans,rot))
            msg.base_pose=self.matrix_to_pose(pose_base_in_world)
            # Query footprint transform from AnymalState message
            suc, pose_footprint_in_world = rc.ros_tf_to_numpy(
                self.query_tf(self.fixed_frame, self.footprint_frame, from_message=anymal_state_msg)
            )
            if not suc:
                self.system_events["state_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_footprint_in_world",
                }
                return
            msg.footprint=self.matrix_to_pose(pose_footprint_in_world)
            self.current_footprint_pose = pose_footprint_in_world
            if self.last_footprint_pose is None:
                self.last_footprint_pose = self.current_footprint_pose
            # Make footprint plane
            footprint_plane = self.make_footprint_with_node(grid_size=10)
            # transform it to geometry_msgs/Point[] format
            msg.footprint_plane.name = "footprint"
            msg.footprint_plane.edge_points = rc.np_to_geometry_msgs_PointArray(footprint_plane)
            self.last_footprint_pose=self.current_footprint_pose
            # Query 4 feet transforms from AnymalState message
            pose_feet_in_world = {}
            foot_poses=[]
            foot_contacts=[]
            foot_planes=[]
            for foot in self.feet_list:
                suc, pose_foot_in_world = rc.ros_tf_to_numpy(
                    self.query_tf(self.fixed_frame, foot, from_message=anymal_state_msg)
                )
                if not suc:
                    self.system_events["state_callback_cancled"] = {
                        "time": rospy.get_time(),
                        "value": f"cancled due to pose_{foot}_in_world",
                    }
                    return
                pose_feet_in_world[foot] = pose_foot_in_world
                foot_pose=self.matrix_to_pose(pose_foot_in_world)
                foot_poses.append(foot_pose)
                # Make feet circle planes
                d=2*self.foot_radius
                foot_plane_points=make_ellipsoid(d,d,0,pose_foot_in_world,grid_size=24)
                foot_plane_points=rc.np_to_geometry_msgs_PointArray(foot_plane_points)
                foot_plane=PlaneEdge()
                foot_plane.edge_points=foot_plane_points
                foot_plane.name=foot
                foot_planes.append(foot_plane)

                # Query each feet contacts from AnymalState message
                for foot_transform in anymal_state_msg.contacts:
                    if foot_transform.name == foot and foot_transform.header.frame_id==self.fixed_frame:
                        foot_contacts.append(foot_transform.state)
                        break
            
            msg.feet_poses=foot_poses
            msg.feet_planes=foot_planes
            msg.feet_contact=foot_contacts
            
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
                fric_pred = torch.clamp(fric_pred, min=0, max=1)
                # Predict using the stiffness predictor
                stiff_pred, self.stiff_hidden = self.stiff_predictor.get_unnormalized_recon(padded_input, self.stiff_hidden)
                stiff_pred = torch.clamp(stiff_pred, min=1, max=10)
            self.input_buffers[0].add(input_data[0].unsqueeze(0))
            # pub fric and stiff together
            new_priv=torch.cat([fric_pred,stiff_pred],dim=-1)
            new_priv=new_priv[:,-1,:].squeeze(0).cpu().numpy()
            msg.prediction=new_priv

            # Publish results
            self.decoder_handler['phy_decoder_pub'].publish(msg)
            if "debug" in self.mode:
                self.visualize_plane(msg)
            
        
        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)
            self.system_events["state_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass


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

    def make_footprint_with_node(self, grid_size: int = 10):
     
        # Get side points
        other_side_points = self.get_side_points(self.last_footprint_pose)
        this_side_points = self.get_side_points(self.current_footprint_pose)
        # swap points to make them counterclockwise
        this_side_points[[0, 1]] = this_side_points[[1, 0]]
        # The idea is to make a polygon like:
        # tsp[1] ---- tsp[0]
        #  |            |
        # osp[0] ---- osp[1]
        # with 'tsp': this_side_points and 'osp': other_side_points
        # Concat points to define the polygon
        points = np.concatenate((this_side_points, other_side_points), axis=0)
        # Make footprint
        # footprint = make_polygon_from_points(points, grid_size=grid_size)
        return points

    def get_side_points(self,pose_footprint_in_world=np.eye(4)):
        return make_plane(x=0.0, y=self.robot_width, pose=pose_footprint_in_world, grid_size=2)

    def visualize_plane(self,msg:PhyDecoderOutput):
        # color.a will be set to 1.0 for foot plane if its contact is true
        suc=False
        header=msg.header
        planes=[x.edge_points for x in msg.feet_planes]
        # planes.append(msg.footprint_plane.edge_points)
        names=[x.name for x in msg.feet_planes]
        # names.append(msg.footprint_plane.name)
        marker_array = MarkerArray()
        for i, plane in enumerate(planes):
            marker=Marker()
            marker.header=header
            marker.ns=names[i]
            marker.type=Marker.LINE_STRIP
            marker.action=Marker.ADD
            marker.scale.x = 0.02
            # uncomment this line if you want to see the plane history
            # marker.id=self.step*len(planes)+i
            marker.lifetime=rospy.Duration(10)
            rgb_color = self.color_palette[i % len(self.color_palette)]
            rgba_color = (rgb_color[0], rgb_color[1], rgb_color[2], 1.0)  # Add alpha value

            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0

            # Set the color of the marker
            marker.color.r = rgba_color[0]
            marker.color.g = rgba_color[1]
            marker.color.b = rgba_color[2]
            marker.color.a = rgba_color[3]
            if names[i] in self.feet_list and msg.feet_contact[i] ==0:
                marker.color.a = 0.1
            else:
                
                marker.points=plane

                marker_array.markers.append(marker)
        self.decoder_handler['marker_planes_pub'].publish(marker_array)
        suc=True
        return suc
 
if __name__ == "__main__":
    node_name = "Phy_decoder_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = PhyDecoder()
    rospy.spin()
