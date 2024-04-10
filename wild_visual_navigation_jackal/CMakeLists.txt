cmake_minimum_required(VERSION 3.0.2)
project(wild_visual_navigation_jackal)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  sensor_msgs
  std_msgs
  wild_visual_navigation_ros
)

catkin_package(
  CATKIN_DEPENDS
)

message(${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY launch Media worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

############
## Python ##
############
catkin_install_python(PROGRAMS scripts/carrot_follower.py
                               scripts/jackal_state_converter_node.py
                               scripts/gazebo_world_publisher.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})