#!/usr/bin/python3
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#

from visualization_msgs.msg import Marker
import rospkg
import rospy


def callback(event):
    pub.publish(marker)


if __name__ == "__main__":
    rospy.init_node("gazebo_world_publisher")

    # Default variables
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("wild_visual_navigation_jackal")
    default_model_file = f"{pkg_path}/Media/models/outdoor.dae"

    marker = Marker()
    marker.header.frame_id = "odom"
    marker.header.stamp = rospy.Time.now()
    marker.id = 0
    marker.ns = "world"
    marker.action = Marker.ADD
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0
    marker.mesh_use_embedded_materials = True
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = f"file://{default_model_file}"

    # Set publisher
    pub = rospy.Publisher("/wild_visual_navigation_jackal/simulation_world", Marker, queue_size=10)

    # Set timer to publish
    rospy.Timer(rospy.Duration(5), callback)
    rospy.loginfo("[gazebo_world_publisher] Published world!")
    rospy.spin()
