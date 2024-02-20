Commands for deployment:

# Start/stop recording
$HOME/workspaces/catkin_ws/src/wild_visual_navigation/wild_visual_navigation_ros/config/recording/start_recording.sh
$HOME/workspaces/catkin_ws/src/wild_visual_navigation/wild_visual_navigation_ros/config/recording/stop_recording.sh

# Copying the data
$HOME/workspaces/catkin_ws/src/anymal_rsl/anymal_rsl/anymal_rsl_utils/anymal_rsl_recording/anymal_rsl_recording/bin/copy_mission_data_from_robot.sh jonfrey dodo 2024-02-06-15-20-33 /Data/2024_02_06_Dodo_MPI_WVN

ssh jonfrey@anymal-dodo-jetson -t 'chown -R jonfrey ~/git/anymal_rsl/anymal_rsl/anymal_rsl_utils/anymal_rsl_recording/anymal_rsl_recording/data/'
ssh jonfrey@anymal-dodo-jetson -t 'chgrp -R jonfrey ~/git/anymal_rsl/anymal_rsl/anymal_rsl_utils/anymal_rsl_recording/anymal_rsl_recording/data/'

$HOME/workspaces/catkin_ws/src/anymal_rsl/anymal_rsl/anymal_rsl_utils/anymal_rsl_recording/anymal_rsl_recording/bin/remove_mission_data_from_robot.sh jonfrey dodo