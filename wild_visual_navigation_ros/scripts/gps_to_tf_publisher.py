import rospy
import tf
from geometry_msgs.msg import PointStamped
import os
import subprocess
import time


def callback(point_stamped, nr):
    br = tf.TransformBroadcaster()
    br.sendTransform(
        (-point_stamped.point.x, -point_stamped.point.y, -point_stamped.point.z),
        (0, 0, 0, 1),
        rospy.Time.now(),
        "map",
        f"sensor{str(nr)}",
    )


root = "/media/Data/Datasets/2022_Perugia/day3/mission_data"
bags = {
    "1": "2022-05-12T09-45-07_mission_0_day_3",
    "2": "2022-05-12T09-57-13_mission_0_day_3",
    "3": "2022-05-12T10-18-16_mission_0_day_3",
    "4": "2022-05-12T10-34-03_mission_0_day_3",
    "5": "2022-05-12T10-45-20_mission_0_day_3",
}

# Launch rosmaster
# roscore_process = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
# roscore_process.kill()
print(" roslaunch wild_visual_navigation_ros trajectory_server.launch")
# trajectory_process = subprocess.Popen(["source /home/jonfrey/catkin_ws/devel/setup.bash && roslaunch wild_visual_navigation_ros trajectory_server.launch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# trajectory_process.kill()
for i, m in bags.items():
    cmd = f"rosbag play {root}/{m}/*gps*.bag -r 10 /rover/piksi/position_receiver_0/ros/pos_enu:=/rover/piksi/position_receiver_0/ros/pos_enu{i}"
    print(cmd)
    # Run simulation
    # try:
    #     subprocess.call(
    #         ["rosbag", "play", f"{root}/{m}/*tf*.bag", "-r", "10", f"/rover/piksi/position_receiver_0/ros/pos_enu:=/rover/piksi/position_receiver_0/ros/pos_enu{i}"],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.DEVNULL,
    #     )rosbag play /media/Data/Datasets/2022_Perugia/day3/mission_data/2022-05-12T09-45-07_mission_0_day_3/*tf*.bag -r 10 /rover/piksi/position_receiver_0/ros/pos_enu:=/rover/piksi/position_receiver_0/ros/pos_enu1
    # except:
    #     print("error")

rospy.init_node("point_to_tf")
sub = rospy.Subscriber("/rover/piksi/position_receiver_0/ros/pos_enu1", PointStamped, callback, 1)
sub = rospy.Subscriber("/rover/piksi/position_receiver_0/ros/pos_enu2", PointStamped, callback, 2)
sub = rospy.Subscriber("/rover/piksi/position_receiver_0/ros/pos_enu3", PointStamped, callback, 3)
sub = rospy.Subscriber("/rover/piksi/position_receiver_0/ros/pos_enu4", PointStamped, callback, 4)
sub = rospy.Subscriber("/rover/piksi/position_receiver_0/ros/pos_enu5", PointStamped, callback, 5)
rospy.spin()
