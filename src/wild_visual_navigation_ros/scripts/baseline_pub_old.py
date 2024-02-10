#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import torch
import torch.nn as nn
import copy
from std_msgs.msg import Float32MultiArray, ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, WrenchStamped
from grid_map_msgs.msg import GridMap
# from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped, PolygonStamped, Point32
import tf
import tf.transformations as tftf
import os
import fnmatch
# from Decoder import BeliefDecoderLightning
from Phy_Decoder import RNNInputBuffer,initialize_models,prepare_padded_input

colors = {
        'nord0': np.array([46, 52, 64]),
        'nord1': np.array([59, 66, 82]),
        'nord2': np.array([67, 76, 94]),
        'nord3': np.array([76, 86, 106]),
        'nord4': np.array([216, 222, 233]),
        'nord5': np.array([229, 233, 240]),
        'nord6': np.array([236, 239, 244]),
        'nord7': np.array([143, 188, 187]),
        'nord8': np.array([136, 192, 208]),
        'nord9': np.array([129, 161, 193]),
        'nord10': np.array([94, 129, 172]),   
        'nord11': np.array([191, 97, 106]),   
        'nord12': np.array([208, 135, 112]),  
        'nord13': np.array([235, 203, 139]),  
        'nord14': np.array([163, 190, 140]),  
        'nord15': np.array([180, 142, 173]),
        'red': np.array([255, 0, 0]),
        'green': np.array([0, 255, 0]),
        }



class HiddenVisualizerNode(object):
    def __init__(self):
        rospy.init_node('hidden_visualizer', anonymous=False)
        batch_size=1
        self.step=0
        self.desired_duration=5000
        self.fric_predictor,self.stiff_predictor,self.cfg=initialize_models()
        self.fric_hidden = self.fric_predictor.init_hidden(batch_size)
        self.stiff_hidden = self.stiff_predictor.init_hidden(batch_size)
        self.seq_length = self.cfg["seq_length"]
        self.input_buffers = {0: RNNInputBuffer()}
        self.feet_labels = ["FOOT_LF", "FOOT_RF", "FOOT_LH", "FOOT_RH"]
        # self.plotter = RealTimePlotter(self.feet_labels,self.desired_duration)

        #  # Set up a flag and QTimer to check the plotter window's visibility
        # self.plotter_ready = False
        # self.check_plotter_timer = QTimer()
        # self.check_plotter_timer.timeout.connect(self.check_plotter_status)
        # self.check_plotter_timer.start(100)  # Check every 100ms

        # # Blocking loop to wait until the plotter window is visible
        # while not self.plotter_ready:
        #     QtWidgets.QApplication.processEvents()  # Allow GUI events to be processed

        self.load_rosparam()
        self.load_model()
        self.friction=[0,0,0,0]
        # init ros
        
        self.map_frame = 'odom'
        self.body_frame = 'base'
        
       
        self.recon_scan = np.zeros(300)
        self.listener = tf.TransformListener()
        self.foot_positions = [np.zeros(3) for i in range(4)]
        self.foot_scan = None
        self.scan_size = 52
        rospy.Subscriber(self.hidden_topic,
                         Float32MultiArray,
                         self.hidden_callback,
                         queue_size=10)
        rospy.Subscriber(self.foot_scan_topic,
                         Marker,
                         self.foot_scan_callback,
                         queue_size=1)

        rospy.Subscriber('/elevation_mapping/elevation_map_recordable',
                         GridMap,
                         self.map_callback,
                         queue_size=1)
        self.scan_publisher = rospy.Publisher('~input_scan',
                                              MarkerArray,
                                              queue_size=1)
        self.scans_publisher = rospy.Publisher('~mean_scans',
                                              MarkerArray,
                                              queue_size=1)
        self.contact_publisher = rospy.Publisher('~contact_states',
                                              MarkerArray,
                                              queue_size=1)
        self.wrench_publisher = rospy.Publisher('~wrench',
                                              MarkerArray,
                                              queue_size=1)
        self.map_publisher = rospy.Publisher('~half_map',
                                              GridMap,
                                              queue_size=1)
        self.freq_pub=rospy.Publisher('/test', Float32, queue_size=10)

        foot_id = ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT']
        self.friction_publishers = [rospy.Publisher('~friction_{}'.format(foot_id[i]),
                                              Float32,
                                              queue_size=1) for i in range(4)]

        self.new_friction_publishers = [rospy.Publisher('~new_friction_{}'.format(foot_id[i]),
                                              Float32,
                                              queue_size=1) for i in range(4)]
        self.new_stiffness_publishers = [rospy.Publisher('~new_stiffness_{}'.format(foot_id[i]),
                                              Float32,
                                              queue_size=1) for i in range(4)]
        
        rospy.Timer(rospy.Duration(0.05), self.timer_callback)

        rospy.spin()

    def load_rosparam(self):
        self.hidden_topic = rospy.get_param('~hidden_topic', '/debug_info')
        self.foot_scan_topic = rospy.get_param('~foot_scan_topic', '/foot_scan')
        self.decoder_name = rospy.get_param('~decoder_name', '/lib/gru_gate_new_decoder_5000_100.pt')

    def load_model(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('anymal_perceptive_inspection')
        model_path = package_path + self.decoder_name
        self.model = torch.jit.load(model_path)
        print('loaded model ', model_path)

    def hidden_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        self.reconstruct(data)

    def foot_scan_callback(self, msg):
        markers = MarkerArray()
        markers_input = MarkerArray()
        # print(msg.points)
        for i, point in enumerate(msg.points):
            m = Marker()
            m.header = msg.header
            m.id = i
            m.type = 1
            m.scale.x = 0.02
            m.scale.y = 0.02
            m.scale.z = 0.02
            m.color.a = 1.0
            # m.color.g = 1.0
            foot_n = i // self.scan_size
            if self.friction[foot_n] > 0.4:
                color = colors['green']
            else:
                color = colors['green']
            m.color.r = color[0] / 255.
            m.color.g = color[1] / 255.
            m.color.b = color[2] / 255.
            

            m.action = 0
            m.pose.position.x = point.x
            m.pose.position.y = point.y
            if self.foot_scan is not None:
                m.pose.position.z = self.foot_scan[i]
            m.pose.orientation.w = 1.0
            markers.markers.append(m)
            # bar = self.variance_bar(
            #         msg.header,
            #         self.foot_scan_min[i],
            #         self.foot_scan_max[i],
            #         point.x, point.y, 1000+i,
            #         )
            # markers.markers.append(bar)
# <<<<<<< HEAD
            m2 = copy.deepcopy(m)
            m2.pose.position.z = point.z
            # m2.color = msg.colors[i]
            color_input = colors['nord11']
            m2.color.a = 1.0
            m2.color.r = color_input[0] / 255.
            m2.color.g = color_input[1] / 255.
            m2.color.b = color_input[2] / 255.

            markers_input.markers.append(m2)
# =======
            # m2 = copy.deepcopy(m)
            # m2.pose.position.z = point.z
            # m2.color = msg.colors[i]
            # markers_input.markers.append(m2)
# >>>>>>> da45cd44ec49314a2cd75e6f041a42ed5d1e1e66
        self.scan_publisher.publish(markers_input)
        self.scans_publisher.publish(markers)

    def variance_bar(self, header, min_h, max_h, x, y, idnum):
        m = Marker()
        m.header = header
        m.id = idnum
        m.type = Marker.CYLINDER
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = max_h - min_h
        m.color.a = 1.0
        m.color.g = 1.0
        m.color.r = 1.0
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = (max_h + min_h) / 2.
        m.pose.orientation.w = 1.0
        return m


    def reconstruct(self, array):
        self.step+=1
        torch_array = torch.from_numpy(array).unsqueeze(0)
        obs, hidden = torch.split(torch_array, [341, 100], dim=1)
        input_data=obs[:,:341]
        # print(input_data[0].device)
        padded_inputs = prepare_padded_input(input_data, self.input_buffers, self.step, 1)    
        padded_input = torch.stack(padded_inputs, dim=0)
        if self.cfg['reset_hidden_each_epoch']:
            self.fric_hidden = self.fric_predictor.init_hidden(1)
            self.stiff_hidden = self.stiff_predictor.init_hidden(1)
        # print('obs ', obs.shape)
        # print('hidden ', hidden.shape)
        with torch.no_grad():
            recons = self.model(obs, hidden)
            # Predict using the friction predictor
            fric_pred, self.fric_hidden = self.fric_predictor.get_unnormalized_recon(padded_input, self.fric_hidden)
            
            # Predict using the stiffness predictor
            stiff_pred, self.stiff_hidden = self.stiff_predictor.get_unnormalized_recon(padded_input, self.stiff_hidden)
        self.input_buffers[0].add(input_data[0].unsqueeze(0))
        if isinstance(fric_pred, torch.Tensor):

            fric_pred = torch.clamp(fric_pred, min=0, max=1)
            stiff_pred = torch.clamp(stiff_pred, min=1, max=10)
            new_priv=torch.cat([fric_pred,stiff_pred],dim=-1)

        else:
            fric_recon_loss = fric_pred[2]
            fric_pred_var=fric_pred[1]
            fric_pred_mean=fric_pred[0]
            fric_pred_mean = torch.clamp(fric_pred_mean, min=0, max=1)
            stiff_recon_loss = stiff_pred[2]
            stiff_pred_var=stiff_pred[1]
            stiff_pred_mean=stiff_pred[0]
            stiff_pred_mean = torch.clamp(stiff_pred_mean, min=1, max=10)
            new_priv=torch.cat([fric_pred_mean,stiff_pred_mean],dim=-1)

        # print("new_priv shape:",new_priv[:,-1,:].squeeze(0).cpu().shape)
        priv = recons[0]
        scan = recons[1]
        scans = recons[3]
        scan = scan.squeeze(0).numpy()
        scans = scans.numpy()
        scan_min = np.min(scans, axis=1)[0]
        scan_max = np.max(scans, axis=1)[0]
        msg=Float32()
        msg.data=1.0
        self.freq_pub.publish(msg)
        # print('scan_min ', scan_min.shape)
        self.foot_scan = self.transform_scan(scan)
        # self.foot_scan_min = self.transform_scan(scan_min)
        # self.foot_scan_max = self.transform_scan(scan_max)
        self.priv_to_marker(priv.squeeze(0).numpy(),new_priv[:,-1,:].squeeze(0).cpu().numpy())
        

    def transform_scan(self, scan):
        foot_scan = scan.copy()
        n_scan = foot_scan.shape[0]
        n = n_scan // 4
        for i in range(4):
            foot_scan[i * n: (i + 1) * n] += self.foot_positions[i][2]
        return foot_scan

    def priv_to_marker(self, priv,new_priv):
        priv_scan = priv[:36]
        contact_state = priv[36:40]
        contact_force = priv[40:52]
        contact_normal = priv[52:64]
        self.friction = priv[64:68]
        ori_pred = {
        "fric": self.friction.reshape(1, 4),
        "stiff": np.ones((1, 4))
        }
        # self.plotter.update_plot(ori_pred)
        # print('friction ', self.friction)
        # print('Shape of self.friction:', self.friction.shape)

        self.new_friction=new_priv[:4]
        self.new_stiffness=new_priv[4:8]
        new_pred = {
        "fric": self.new_friction.reshape(1, 4),
        "stiff": self.new_stiffness.reshape(1, 4)
        }
        thigh_shank_contact = priv[68:76]
        external_force = priv[76:79]
        external_torque = priv[79:82]
        air_time = priv[82:86]
        self.contact_marker(contact_state, contact_force, contact_normal)
        self.external_wrench_marker(external_force, external_torque)
        self.publish_friction(self.friction,self.new_friction,self.new_stiffness)

    def publish_friction(self, friction,new_friction,new_stiffness):
        for i in range(4):
            self.friction_publishers[i].publish(friction[i])
            self.new_friction_publishers[i].publish(new_friction[i])
            self.new_stiffness_publishers[i].publish(new_stiffness[i])

    def contact_marker(self, contact_state, contact_force, contact_normal):
        markers = MarkerArray()
        scale = Vector3(0.03, 0.05, 0.05)
        color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        contact_force = contact_force.reshape(4, 3)
        # print('contact state ', contact_state)
        contact_normal = contact_normal.reshape(4, 3)
        for i, state in enumerate(contact_state):
            if state > 0.5:
                tail = Point(*self.foot_positions[i].tolist())
                force_in_map = np.dot(self.body_R, contact_force[i]) * 0.005
                normal_in_map = np.dot(self.body_R, contact_normal[i]) * 0.5
                tip = self.foot_positions[i] + force_in_map
                tip = Point(*tip.tolist())
                marker = self.make_arrow_points_marker(scale,
                                                       color,
                                                       tail,
                                                       tip,
                                                       i)
                markers.markers.append(marker)
                # tip2 = self.foot_positions[i] + normal_in_map
                # tip2 = Point(*tip2.tolist())
                # marker2 = self.make_arrow_points_marker(scale,
                #                                        ColorRGBA(0.0, 1.0, 1.0, 1.0),
                #                                        tail,
                #                                        tip2,
                #                                        i * 10)
                # markers.markers.append(marker2)

                                                       
        # self.scan_publisher.publish(msg)
        self.contact_publisher.publish(markers)

    def external_wrench_marker(self, external_force, external_torque):
        markers = MarkerArray()
        scale = Vector3(0.03, 0.05, 0.05)
        color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        # force
        force_in_map = np.dot(self.body_R, external_force) * 0.200
        torque_in_map = np.dot(self.body_R, external_torque) * 0.200

        tail = Point(*self.body_t.tolist())

        tip_f = self.body_t + force_in_map
        tip_f = Point(*tip_f.tolist())
        marker = self.make_arrow_points_marker(scale,
                                               color,
                                               tail,
                                               tip_f,
                                               100)
        markers.markers.append(marker)

        # color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        # tip_t = self.body_t + torque_in_map
        # tip_t = Point(*tip_t.tolist())
        # marker = self.make_arrow_points_marker(scale,
        #                                        color,
        #                                        tail,
        #                                        tip_t,
        #                                        1)
        # markers.markers.append(marker)
        self.wrench_publisher.publish(markers)
        # print('external_force ', force_in_map)
        # print('external_torque ', torque_in_map)
        # force_in_map = np.dot(self.body_R, external_force) * 0.100
        # torque_in_map = np.dot(self.body_R, external_torque) * 0.100
        # print('external_force ', force_in_map)
        # print('external_torque ', torque_in_map)
        # markers = MarkerArray()
        # scale = Vector3(0.03, 0.05, 0.05)
        # color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        # tail = Point(*self.body_t.tolist())
        # tip = self.body_t + force_in_map
        # tip = Point(*tip.tolist())
        # marker = self.make_arrow_points_marker(scale,
        #                                        ColorRGBA(1.0, 0.0, 0.0, 1.0),
        #                                        tail,
        #                                        tip,
        #                                        0)
        # markers.markers.append(marker)
        # tip2 = self.body_t + torque_in_map
        # tip2 = Point(*tip2.tolist())
        # marker = self.make_arrow_points_marker(scale,
        #                                        ColorRGBA(0.0, 0.0, 1.0, 1.0),
        #                                        tail,
        #                                        tip2,
        #                                        1)
        # markers.markers.append(marker)
        # self.wrench_publisher.publish(markers)
        # wrench = WrenchStamped()
        # wrench.header.frame_id = self.body_frame
        # wrench.header.stamp = rospy.Time.now()
        # wrench.wrench.force = Vector3(*force_in_map.tolist())
        # wrench.wrench.torque = Vector3(*torque_in_map.tolist())
        # self.wrench_publisher.publish(wrench)

    def make_arrow_points_marker(self, scale, color, tail, tip, idnum):
        # make a visualization marker array for the occupancy grid
        m = Marker()
        m.action = Marker.ADD
        m.header.frame_id = self.map_frame
        m.header.stamp = rospy.Time.now()
        m.ns = 'points_arrows'
        m.id = idnum
        m.lifetime = rospy.Duration(0.05)
        m.type = Marker.ARROW
        m.pose.orientation.y = 0
        m.pose.orientation.w = 1
        m.scale = scale
        m.color = color
        m.points = [ tail, tip ]
        return m

    def update_base_tf(self):
        try:
            self.listener.waitForTransform(self.map_frame,
                                           self.body_frame,
                                           rospy.Time(0),
                                           rospy.Duration(0.1))
            transform = self.listener.lookupTransform(self.map_frame,
                                                      self.body_frame,
                                                      rospy.Time(0))
            translation, quaternion = transform
            self.body_t = np.array(translation)
            self.body_R = tftf.quaternion_matrix(quaternion)[0:3, 0:3]
        except (tf.Exception) as exc:
            rospy.logerr("tf error when resolving tf: " + str(exc))
            return

    def update_foot_tf(self):
        foot_frames = ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT']
        for i in range(4):
            try:
                self.listener.waitForTransform(self.map_frame,
                                               foot_frames[i],
                                               rospy.Time(0),
                                               rospy.Duration(0.1))
                transform = self.listener.lookupTransform(self.map_frame,
                                                          foot_frames[i],
                                                          rospy.Time(0))
                translation, quaternion = transform
                self.foot_positions[i] = np.array(translation)
            except (tf.Exception) as exc:
                rospy.logerr("tf error when resolving tf: " + str(exc))
                return

    def timer_callback(self, e):
        # QtWidgets.QApplication.processEvents()  # Process PyQt events
        self.update_base_tf()
        self.update_foot_tf()

    def map_callback(self, msg):
        # print('got map', len(msg.data), len(msg.data[0].data))
        data = np.array(msg.data[0].data)
        # print(data.shape)
        dim = int(np.sqrt(data.shape[0]))
        data = data.reshape(dim, dim)
        # print(data.shape)
        data[:dim//3, :] = np.nan
        data[2*dim//3:, :] = np.nan
        data[:, :dim//3] = np.nan
        data[:, 2*dim//3:] = np.nan
        # print(data)
        msg.data[0].data = data.flatten()
        self.map_publisher.publish(msg)


if __name__ == '__main__':
    try:
        node = HiddenVisualizerNode()
    except rospy.ROSInterruptException:
        pass
