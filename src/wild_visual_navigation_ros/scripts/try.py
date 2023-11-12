import rospy
import tf
import tf2_ros
import numpy as np
import ros_converter as rc
from geometry_msgs.msg import TransformStamped

def publish_transform():
    rospy.init_node('tf_broadcaster')

    # Create a single TF2 broadcaster
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Define the first static transform from 'base' to 'hdr_base'
    static_transformStamped = TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "base"
    static_transformStamped.child_frame_id = "hdr_base"
    static_transformStamped.transform.translation.x = 0.3537
    static_transformStamped.transform.translation.y = 0.0
    static_transformStamped.transform.translation.z = 0.1634

    # Assuming a quaternion is provided here
    static_transformStamped.transform.rotation.x = 0.997
    static_transformStamped.transform.rotation.y = 0.0
    static_transformStamped.transform.rotation.z = -0.072
    static_transformStamped.transform.rotation.w = 0.0
    

   

    static_transformStamped_2 = TransformStamped()
    # Define the second static transform from 'hdr_base' to 'hdr_cam'
    static_transformStamped_2.header.stamp = rospy.Time.now()
    static_transformStamped_2.header.frame_id = "hdr_base"
    static_transformStamped_2.child_frame_id = "hdr_cam"
    static_transformStamped_2.transform.translation.x = 0.0
    static_transformStamped_2.transform.translation.y = 0.0
    static_transformStamped_2.transform.translation.z = 0.0

    # Convert Euler angles to a quaternion
    roll, pitch, yaw = -1.5708, 0.0, -1.5708
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    static_transformStamped_2.transform.rotation.x = quaternion[0]
    static_transformStamped_2.transform.rotation.y = quaternion[1]
    static_transformStamped_2.transform.rotation.z = quaternion[2]
    static_transformStamped_2.transform.rotation.w = quaternion[3]

    T_0=np.array([[-1,0,0,0],
                  [0,-1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    suc, T_1 = rc.ros_tf_to_numpy(((0.3537,0.0,0.1634), (0.997,0.0,-0.072,0.0)))
    T_f=T_0@T_1
    print(T_f)
    # T_f=np.array([[-0.98962361,  0.        , -0.14368394,  0.3537],
    #             [ 0.         , 0.99999982,  0.         , 0.        ],
    #             [ 0.14368394 , 0.        , -0.98962343 , 0.1634],
    #             [ 0.          ,0.         , 0.         , 1.        ]])
    from scipy.spatial.transform import Rotation as R
    # Extract the translation (last column of the transformation matrix)
    translation = T_f[0:3, 3]

    # Extract the rotation matrix (top-left 3x3 submatrix of the transformation matrix)
    rotation_matrix = T_f[0:3, 0:3]

    # Convert the rotation matrix to a quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns (x, y, z, w)

    print("Translation:", translation)
    print("Quaternion:", quaternion) 

    static_transformStamped_3 = TransformStamped()
    static_transformStamped_3.header.stamp = rospy.Time.now()
    static_transformStamped_3.header.frame_id = "base"
    static_transformStamped_3.child_frame_id = "hdr_rear_base"
    static_transformStamped_3.transform.translation.x = -0.3537
    static_transformStamped_3.transform.translation.y = 0.0
    static_transformStamped_3.transform.translation.z = 0.1634

    # Assuming a quaternion is provided here
    static_transformStamped_3.transform.rotation.x = 0
    static_transformStamped_3.transform.rotation.y = 0.997
    static_transformStamped_3.transform.rotation.z = 0
    static_transformStamped_3.transform.rotation.w = 0.072

    static_transformStamped_4 = TransformStamped()
    # Define the second static transform from 'hdr_base' to 'hdr_cam'
    static_transformStamped_4.header.stamp = rospy.Time.now()
    static_transformStamped_4.header.frame_id = "hdr_rear_base"
    static_transformStamped_4.child_frame_id = "hdr_rear_cam"
    static_transformStamped_4.transform.translation.x = 0.0
    static_transformStamped_4.transform.translation.y = 0.0
    static_transformStamped_4.transform.translation.z = 0.0

    # Convert Euler angles to a quaternion
    roll, pitch, yaw = -1.5708, 0.0, -1.5708
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    static_transformStamped_4.transform.rotation.x = quaternion[0]
    static_transformStamped_4.transform.rotation.y = quaternion[1]
    static_transformStamped_4.transform.rotation.z = quaternion[2]
    static_transformStamped_4.transform.rotation.w = quaternion[3]


    # Broadcast the second transform
    broadcaster.sendTransform([static_transformStamped, static_transformStamped_2, static_transformStamped_3, static_transformStamped_4])
    rospy.loginfo('Broadcasting: hdr_base to hdr_cam transform')

    # Spin to keep the script from exiting
    rospy.spin()

if __name__ == '__main__':
    a=np.array([[1,2,3],[4,5,6],[7,8,9]])
    b=a[:,2]
    print(a[:,2])
    try:
        publish_transform()
    except rospy.ROSInterruptException:
        pass
