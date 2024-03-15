#!/bin/bash
# play_rosbags.sh

# Navigate to the directory containing the bag files
# cd /media/chen/UDisk1/vis_rosbag/snow
# cd /media/chen/Chen/rosbag_white/xxxxx9th
cd /media/chen/Chen/2024-01-25-white-board/7th
# cd /media/chen/UDisk1/vis_rosbag/single_test
# cd /media/chen/UDisk1/vis_rosbag/val
# cd /media/chen/UDisk1/vis_rosbag/snow/val
# Find all bag files and play them
# cd /media/chen/UDisk1/vis_rosbag/new_snow
# cd /media/chen/UDisk1/vis_rosbag/lab
# cd /media/chen/Chen/rosbag_lee

# 10cm foam
# cd /media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-13-22-53/mission_data

# 5cm foam
# cd /media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-13-17-57/mission_data

# 10cm foam 2 (closer shot) 320  5:25-29 in video
# cd /media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-13-35-49/mission_data

# step white 434 first 28s in csv
# cd /media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-15-53-53/mission_data

# walk white
# cd /media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-14-28-25/mission_data

rosbag play --clock *.bag -r 0.5

#roslaunch wild_visual_navigation_ros play.launch
# /wide_angle_camera_rear/image_color_rect

# Snow ground 1670683273.696940
# 