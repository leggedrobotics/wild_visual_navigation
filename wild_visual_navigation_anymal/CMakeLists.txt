#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
cmake_minimum_required(VERSION 3.0.2)
project(wild_visual_navigation_anymal)
set (CMAKE_CXX_STANDARD 14)

set(CATKIN_PACKAGE_LIST 
  rospy
  roscpp
  sensor_msgs
  std_msgs
  wild_visual_navigation_msgs
)

find_package(catkin REQUIRED 
  COMPONENTS
    ${CATKIN_PACKAGE_LIST}
)

catkin_package(
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_LIST}
)

###########
## Build ##
###########
include_directories(
  SYSTEM
    ${catkin_INCLUDE_DIRS}
)

if(BUILD_ANYMAL)
   find_package( anymal_msgs )
   # Declare node
   add_executable( anymal_msg_converter_cpp_node
                   src/anymal_msg_converter_cpp_node.cpp)

   target_link_libraries(anymal_msg_converter_cpp_node
                      ${catkin_LIBRARIES}
   )
endif()

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${TARGETS}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY
    include/${PROJECT_NAME}/
  DESTINATION
    ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

############
## Python ##
############
catkin_install_python(PROGRAMS scripts/anymal_msg_converter_node.py
                               scripts/policy_debug_info_converter_node.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})