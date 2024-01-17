# Contains header, prediction array, 3d feet coordinates in odom frame
# and 3d footprint coordinates in odom frame and their plane info
# and contact info for each feet
std_msgs/Header header
geometry_msgs/Pose base_pose
float32[] prediction
float32[] prediction_var
float32[] recon_loss
geometry_msgs/Pose[] feet_poses
bool[] feet_contact
wild_visual_navigation_msgs/PlaneEdge[] feet_planes
geometry_msgs/Pose footprint
wild_visual_navigation_msgs/PlaneEdge footprint_plane




