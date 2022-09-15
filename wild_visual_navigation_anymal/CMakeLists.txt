cmake_minimum_required(VERSION 3.0.2)
project(wild_visual_navigation_anymal)

find_package(catkin REQUIRED COMPONENTS
  rospy
  anymal_msgs
  sensor_msgs
  std_msgs
  wild_visual_navigation_msgs
)

catkin_package(
  CATKIN_DEPENDS
)

catkin_install_python(PROGRAMS scripts/anymal_msg_converter_node.py
                               scripts/policy_debug_info_converter_node.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
