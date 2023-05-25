#!/bin/bash

args=""
for option in "$@"; do
  if [ "$option" == "--sem" ]; then
    args="$args /elevation_mapping/elevation_map_raw:=/recorded/elevation_mapping/elevation_map_raw \
    /elevation_mapping/semantic_map_raw:=/recorded/elevation_mapping/semantic_map_raw"
  elif [ "$option" == "--wvn" ]; then
    args="$args /wild_visual_navigation_node/front/camera_info:=/recorded_wvn/wild_visual_navigation_node/front/camera_info \
    /wild_visual_navigation_node/front/confidence:=/recorded_wvn/wild_visual_navigation_node/front/confidence \
    /wild_visual_navigation_node/front/image_input:=/recorded_wvn/wild_visual_navigation_node/front/image_input \
    /wild_visual_navigation_node/front/traversability:=/recorded_wvn/wild_visual_navigation_node/front/traversability \
    /wild_visual_navigation_node/graph_footprints:=/recorded_wvn/wild_visual_navigation_node/graph_footprints \
    /wild_visual_navigation_node/instant_traversability:=/recorded_wvn/wild_visual_navigation_node/instant_traversability \
    /wild_visual_navigation_node/proprioceptive_graph:=/recorded_wvn/wild_visual_navigation_node/proprioceptive_graph \
    /wild_visual_navigation_node/robot_state:=/recorded_wvn/wild_visual_navigation_node/robot_state \
    /wild_visual_navigation_node/system_state:=/recorded_wvn/wild_visual_navigation_node/system_state \
    /wild_visual_navigation_visu_confidence/confidence_overlayed:=/recorded_wvn/wild_visual_navigation_visu_confidence/confidence_overlayed \
    /wild_visual_navigation_visu_traversability/traversability_overlayed:=/recorded_wvn/wild_visual_navigation_visu_traversability/traversability_overlayed"
  elif [ "$option" == "--flp" ]; then
    args="$args /field_local_planner/action_server/status:=/recorded_flp/field_local_planner/action_server/status \
    /field_local_planner/current_base:=/recorded_flp/field_local_planner/current_base \
    /field_local_planner/current_goal:=/recorded_flp/field_local_planner/current_goal \
    /field_local_planner/parameter_descriptions:=/recorded_flp/field_local_planner/parameter_descriptions \
    /field_local_planner/parameter_updates:=/recorded_flp/field_local_planner/parameter_updates \
    /field_local_planner/path:=/recorded_flp/field_local_planner/path \
    /field_local_planner/real_carrot:=/recorded_flp/field_local_planner/real_carrot \
    /field_local_planner/rmp/control_points:=/recorded_flp/field_local_planner/rmp/control_points \
    /field_local_planner/rmp/parameter_descriptions:=/recorded_flp/field_local_planner/rmp/parameter_descriptions \
    /field_local_planner/rmp/parameter_updates:=/recorded_flp/field_local_planner/rmp/parameter_updates \
    /field_local_planner/status:=/recorded_flp/field_local_planner/status \
    /elevation_mapping/elevation_map_wifi:=/recorded_flp/elevation_mapping/elevation_map_wifi"
  elif [ "$option" == "--tf" ]; then
    args="$args /tf:=/recorded_flp/tf" 
    
    # /tf_static:=/recorded_flp/tf_static"

    echo "rosrun anymal_rsl_launch replay.py c /media/Data/Datasets/2023_Oxford_Testing/2023_01_27_Oxford_Park/mission_data/2023-01-27-11-00-22/2023-01-27-11-00-22_anymal-coyote-lpc_mission.yaml"

  elif [ "$option" == "--compslam" ]; then
    args="$args /compslam_lio/odometry:=/recorded_comslam/compslam_lio/odometry \
    /msf_compslam_lio_body_imu/msf_core/odometry:=/recorded_comslam/msf_compslam_lio_body_imu/msf_core/odometry \
    /loam/map:=/recorded_comslam/loam/map \
    /loam/odometry:=/recorded_comslam/loam/odometry"

  else
    args="$args $option"
  fi
done

echo args
rosparam set use_sim_time true
rosbag play --clock $args
