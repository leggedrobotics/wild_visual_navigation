#!/bin/bash
# play_rosbags.sh

# Navigate to the directory containing the bag files
# cd /media/chen/UDisk1/vis_rosbag/snow
cd /media/chen/UDisk1/vis_rosbag/single_test
# cd /media/chen/UDisk1/vis_rosbag/val
# cd /media/chen/UDisk1/vis_rosbag/snow/val
# Find all bag files and play them
# cd /media/chen/UDisk1/vis_rosbag/new_snow
# cd /media/chen/UDisk1/vis_rosbag/lab
# cd /media/chen/Chen/rosbag_lee
rosbag play --clock *.bag

#roslaunch wild_visual_navigation_ros play.launch
# /wide_angle_camera_rear/image_color_rect