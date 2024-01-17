cmake_minimum_required(VERSION 3.0.2)
project(footcontact_history)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs tf2_ros)
catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS scripts/footcontact_history_node.py DESTINATION
                      ${CATKIN_PACKAGE_BIN_DESTINATION})
