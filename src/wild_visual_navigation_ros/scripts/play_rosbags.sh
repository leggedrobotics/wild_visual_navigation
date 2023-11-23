#!/bin/bash
# play_rosbags.sh

# Navigate to the directory containing the bag files
# cd /media/chen/UDisk1/vis_rosbag/snow
cd /media/chen/UDisk1/vis_rosbag/single_test

# Find all bag files and play them

rosbag play --clock *.bag

#roslaunch wild_visual_navigation_ros play.launch
