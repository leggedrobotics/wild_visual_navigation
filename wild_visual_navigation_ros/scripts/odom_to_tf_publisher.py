import rospy
import tf
from geometry_msgs.msg import PoseStamped

i = 0
def pose_callback(data):
    br = tf.TransformBroadcaster()
    br.sendTransform((data.pose.position.x, data.pose.position.y, data.pose.position.z),
                     (data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w),
                     data.header.stamp,
                     "base",
                     "abc")
    print(data.header.frame_id)
    print(data.header.stamp)
    print("published")

if __name__ == '__main__':
    rospy.init_node('pose_to_tf_node_new')
    rospy.Subscriber("/mapping_node/o3d_slam_lidar_pose_in_map", PoseStamped, pose_callback)
    rospy.spin()