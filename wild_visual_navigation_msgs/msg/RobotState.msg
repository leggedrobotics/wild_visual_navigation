# Header
std_msgs/Header header

# Standard robot states
# We consider that any robot should have access to 6DoF pose and twist estimates
geometry_msgs/PoseStamped pose
geometry_msgs/TwistStamped twist


# Other custom states
# Here we can store an arbitrary number of states, described by a CustomState msg
wild_visual_navigation_msgs/CustomState[] states
