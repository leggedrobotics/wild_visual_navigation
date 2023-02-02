import rospy
from nav_msgs.msg import Path

import numpy as np


def callback(data, args):
    (topic_name,) = args
    path_length = 0
    for i in range(len(data.poses) - 1):
        p1 = np.array([data.poses[i].pose.position.x, data.poses[i].pose.position.y, data.poses[i].pose.position.z])
        p2 = np.array(
            [data.poses[i + 1].pose.position.x, data.poses[i + 1].pose.position.y, data.poses[i + 1].pose.position.z]
        )
        path_length += np.linalg.norm(p2 - p1)
    print("Received path on topic {} with length {} meters".format(topic_name, path_length))


if __name__ == "__main__":
    rospy.init_node("path_subscriber")
    topic_names = ["/gps_trajectory1", "/gps_trajectory2", "/gps_trajectory3", "/gps_trajectory4", "/gps_trajectory5"]
    for topic_name in topic_names:
        rospy.Subscriber(topic_name, Path, callback, callback_args=(topic_name,))
    rospy.spin()
