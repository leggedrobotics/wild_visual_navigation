#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
cmake_minimum_required(VERSION 3.0.2)
project(wild_visual_navigation_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  sensor_msgs
  std_msgs
  wild_visual_navigation_msgs
)

catkin_package(
  CATKIN_DEPENDS
)

catkin_python_setup()
catkin_install_python(PROGRAMS  scripts/wvn_feature_extractor_node.py
                                scripts/wvn_learning_node.py
                                scripts/overlay_images.py
                                scripts/smart_carrot.py
                                scripts/rosbag_play.sh
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
