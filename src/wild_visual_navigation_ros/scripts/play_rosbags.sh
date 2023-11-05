#!/bin/bash
# play_rosbags.sh

# Navigate to the directory containing the bag files
cd /media/chen/UDisk/vis_rosbag

# Find all bag files and play them

rosbag play --clock *.bag

#roslaunch wild_visual_navigation_ros play.launch
