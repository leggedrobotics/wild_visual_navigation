cmake_minimum_required(VERSION 3.0.2)
project(wild_visual_navigation_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  anymal_msgs
  nav_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
)

catkin_python_setup()
catkin_install_python(PROGRAMS scripts/demo_ros_node.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})