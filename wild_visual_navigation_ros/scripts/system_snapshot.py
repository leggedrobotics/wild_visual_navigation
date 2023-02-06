from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation_msgs.msg import SystemState
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker

import cv2
from cv_bridge import CvBridge

import tf2_ros
import message_filters
import rospy
import sys
import os
import numpy as np
import math


class SystemSnapshot:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/wild_visual_navigation_node/front/image_input")
        self.trav_topic = rospy.get_param(
            "~traversability_topic", "/wild_visual_navigation_visu_traversability/traversability_overlayed"
        )
        self.conf_topic = rospy.get_param(
            "~confidence_topic", "/wild_visual_navigation_visu_confidence/confidence_overlayed"
        )
        self.state_topic = rospy.get_param("~state_topic", "/wild_visual_navigation_node/system_state")

        self.base_frame = rospy.get_param("~base_frame", "base")
        self.fixed_frame = rospy.get_param("~fixed_frame", "point_cloud_odom")
        self.distance_thr = rospy.get_param("~distance_thr", 1.5)  # meters
        self.ignore_z = rospy.get_param("~ignore_z_for_distance", True)  # Ignore z when computing distance

        # Snapshot index
        self.snapshot_idx = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.bridge = CvBridge()
        self.snapshot_service = rospy.Service("~trigger", Trigger, self.snapshot_request_callback)
        self.marker_publisher = rospy.Publisher("~snapshots_markers", MarkerArray, queue_size=10)

        input_sub = message_filters.Subscriber(self.input_topic, Image)
        trav_sub = message_filters.Subscriber(self.trav_topic, Image)
        conf_sub = message_filters.Subscriber(self.conf_topic, Image)
        state_sub = message_filters.Subscriber(self.state_topic, SystemState)

        ts = message_filters.ApproximateTimeSynchronizer(
            [input_sub, trav_sub, conf_sub, state_sub], 10, slop=0.5, allow_headerless=True
        )
        ts.registerCallback(self.callback)

        self.last_data = {}
        self.snapshots = {}
        self.snapshot_markers = MarkerArray()

    def callback(self, input_msg, trav_msg, conf_msg, state_msg):
        # Get step number
        step = f"{state_msg.step}"
        loss_total = f"{state_msg.loss_total:.4f}"
        loss_trav = f"{state_msg.loss_trav:.4f}"
        loss_reco = f"{state_msg.loss_reco:.4f}"
        thr = f"{state_msg.scale_traversability_threshold:.4f}"
        stamp = f"{input_msg.header.stamp.to_sec():.4f}"

        # Prepare string to save files
        learning_str = f"step_{step}_stamp_{stamp}_loss_total_{loss_total}_trav_{loss_trav}_reco_{loss_reco}_thr_{thr}"

        # Convert images to numpy
        cv_input = self.bridge.imgmsg_to_cv2(input_msg, desired_encoding="bgr8")
        cv_trav = self.bridge.imgmsg_to_cv2(trav_msg, desired_encoding="bgr8")
        cv_conf = self.bridge.imgmsg_to_cv2(conf_msg, desired_encoding="bgr8")

        self.last_data = {"input": cv_input, "trav": cv_trav, "conf": cv_conf, "learning": learning_str, "stamp": stamp}

        # Get pose
        pos, rot = self.query_tf(self.fixed_frame, self.base_frame)

        # Check against all the snapshots
        if self.snapshots:
            for i in self.snapshots:
                v = self.snapshots[i]
                # Compute distance
                d = (pos[0] - v["position"][0]) ** 2 + (pos[1] - v["position"][1]) ** 2
                d += 0.0 if self.ignore_z else (pos[2] - v["position"][2]) ** 2
                d = math.sqrt(d)
                if d < self.distance_thr:
                    self.store_snapshot(i)

        self.marker_publisher.publish(self.snapshot_markers)

    def store_snapshot(self, idx):
        input_folder = os.path.join(WVN_ROOT_DIR, f"results/snapshots/{idx}/input")
        trav_folder = os.path.join(WVN_ROOT_DIR, f"results/snapshots/{idx}/trav")
        conf_folder = os.path.join(WVN_ROOT_DIR, f"results/snapshots/{idx}/conf")

        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(trav_folder, exist_ok=True)
        os.makedirs(conf_folder, exist_ok=True)

        try:
            suffix = self.last_data["learning"]
            cv2.imwrite(f"{input_folder}/{suffix}.png", self.last_data["input"])
            cv2.imwrite(f"{trav_folder}/{suffix}.png", self.last_data["trav"])
            cv2.imwrite(f"{conf_folder}/{suffix}.png", self.last_data["conf"])

            print(f"Saving snapshot {idx}/{suffix}")
            return (True, f"{idx}/{suffix}")
        except Exception as e:
            return (False, e)

    def snapshot_request_callback(self, req):
        print(f"Current idx {self.snapshot_idx}")
        # Query pose
        pos, rot = self.query_tf(self.fixed_frame, self.base_frame)

        # Add snapshot point
        self.snapshots[self.snapshot_idx] = {"position": pos}

        # Save snapshot
        success, msg = self.store_snapshot(self.snapshot_idx)

        # Update markers
        self.update_markers(self.snapshot_idx, pos)
        self.marker_publisher.publish(self.snapshot_markers)

        # Update index
        self.snapshot_idx += 1

        return TriggerResponse(success=success, message=msg)

    def update_markers(self, idx, pos):
        marker = Marker()
        marker.header.frame_id = self.fixed_frame
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.frame_locked = False
        marker.ns = "snapshots"
        self.snapshot_markers.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = self.fixed_frame
        marker.id = 100 + idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0] + 1.0
        marker.pose.position.y = pos[1] + 1.0
        marker.pose.position.z = pos[2]
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.text = f"{idx}"
        marker.frame_locked = False
        marker.ns = "snapshots"
        self.snapshot_markers.markers.append(marker)

    def query_tf(self, parent_frame: str, child_frame: str, stamp=None):
        if stamp is None:
            stamp = rospy.Time(0)
        try:
            res = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp, timeout=rospy.Duration(0.03))
            trans = (res.transform.translation.x, res.transform.translation.y, res.transform.translation.z)
            rot = np.array(
                [res.transform.rotation.x, res.transform.rotation.y, res.transform.rotation.z, res.transform.rotation.w]
            )
            rot /= np.linalg.norm(rot)
            return (trans, tuple(rot))
        except Exception as e:
            print("Error in query tf: ", e)
            rospy.logwarn(f"Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)


if __name__ == "__main__":
    print("MAIN NAME")
    try:
        rospy.init_node("wild_visual_navigation_snapshot")
    except:
        nr = "_" + rospy.myargv(argv=sys.argv)[-1].split(" ")[-1]
        rospy.init_node(f"wild_visual_navigation_snapshot_{nr}")
    print("Start")
    wvn = SystemSnapshot()
    rospy.spin()
