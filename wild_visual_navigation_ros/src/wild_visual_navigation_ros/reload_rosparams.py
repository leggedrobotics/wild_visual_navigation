#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import rospkg
import rospy
import os

def reload_rosparams(enabled, node_name, camera_cfg):
    if enabled:
        try:
            rospy.delete_param(node_name)
        except:
            pass
        
        rospack = rospkg.RosPack()
        wvn_path = rospack.get_path("wild_visual_navigation_ros")
        os.system(f"rosparam load {wvn_path}/config/wild_visual_navigation/default.yaml {node_name}")
        wvn_anymal = rospack.get_path("wild_visual_navigation_anymal")
        os.system(f"rosparam load {wvn_anymal}/config/wild_visual_navigation/inputs/{camera_cfg}.yaml {node_name}")
