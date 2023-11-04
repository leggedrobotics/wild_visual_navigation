# Contains header, prediction array, 3d feet coordinates in odom frame
# and 3d footprint coordinates in odom frame and their plane info
# and contact info for each feet
std_msgs/Header header
float32[] prediction
geometry_msgs/Pose[] feet_poses
bool[] feet_contact
wild_visual_navigation_msgs/PlaneEdge[] feet_planes
geometry_msgs/Pose footprint
wild_visual_navigation_msgs/PlaneEdge footprint_plane




