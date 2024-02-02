import rospy
import tf
import tf2_ros
import numpy as np
import ros_converter as rc
from geometry_msgs.msg import TransformStamped
from msg_to_transmatrix import pq_to_se3
class StaticTransform():
    def __init__(self,translation,quat,parent,child) -> None:
        
        self.static_transformStamped = TransformStamped()
        self.static_transformStamped.header.stamp=rospy.Time.now()
        self.static_transformStamped.header.frame_id=parent
        self.static_transformStamped.child_frame_id=child
        self.static_transformStamped.transform.translation.x=translation[0]
        self.static_transformStamped.transform.translation.y=translation[1]
        self.static_transformStamped.transform.translation.z=translation[2]
        
        self.static_transformStamped.transform.rotation.x=quat[0]
        self.static_transformStamped.transform.rotation.y=quat[1]
        self.static_transformStamped.transform.rotation.z=quat[2]
        self.static_transformStamped.transform.rotation.w=quat[3]
        
        self.transformation_matrix=pq_to_se3(translation,quat)
        self.parent=parent
        self.child=child
        self.translation=np.array(translation)
        self.quat=np.array(quat)
    
    def create_from_matrix(self,matrix:np.ndarray)-> 'StaticTransform':
        if matrix.shape!=(4,4):
            raise ValueError("Matrix must be 4x4")
        translation=matrix[:3,3]
        quat=tf.transformations.quaternion_from_matrix(matrix)
        return StaticTransform(translation,quat,self.parent,self.child)
    
    def multiply(self,other:'StaticTransform') -> 'StaticTransform':
        
        pass
        
        
        
        
        

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
    
    x, y, z = 0.3537, 0.0, 0.1634
    translation_matrix = tf.transformations.translation_matrix((x, y, z))
    rotation_matrix_in_parent_wf = tf.transformations.quaternion_matrix((0.997,0.0,-0.072,0.0))
    rotation_matrix_in_parent_wf=translation_matrix@rotation_matrix_in_parent_wf
   

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

    x,y,z=0.0,0.0,0.0
    translation_matrix = tf.transformations.translation_matrix((x, y, z))
    roll, pitch, yaw = -1.5708, 0.0, -1.5708
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    rotation_matrix_in_parent_fo = tf.transformations.quaternion_matrix(quaternion)
    rotation_matrix_in_parent_fo=translation_matrix@rotation_matrix_in_parent_fo
    
    ok=rotation_matrix_in_parent_wf@rotation_matrix_in_parent_fo
    print("hdr front")
    print(ok)
    
    T_0=np.array([[-1,0,0,0],
                  [0,-1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    suc, T_1 = rc.ros_tf_to_numpy(((0.3537,0.0,0.1634), (0.997,0.0,-0.072,0.0)))
    T_f=T_0@T_1
    print(T_f)
    # T_f=np.array([[ 3.63509049e-06, -1.43680305e-01, -9.89624138e-01,
    #     -3.53700000e-01],
    #    [-9.99999820e-01,  1.34923159e-11, -3.67320444e-06,
    #      0.00000000e+00],
    #    [ 5.27780582e-07,  9.89623958e-01, -1.43680305e-01,
    #      1.63400000e-01],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]])
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
    static_transformStamped_3.transform.rotation.x = quaternion[0]
    static_transformStamped_3.transform.rotation.y = quaternion[1]
    static_transformStamped_3.transform.rotation.z = quaternion[2]
    static_transformStamped_3.transform.rotation.w = quaternion[3]

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
    rotation_matrix_in_parent = tf.transformations.quaternion_matrix(quaternion)
    static_transformStamped_4.transform.rotation.x = quaternion[0]
    static_transformStamped_4.transform.rotation.y = quaternion[1]
    static_transformStamped_4.transform.rotation.z = quaternion[2]
    static_transformStamped_4.transform.rotation.w = quaternion[3]

    res=T_f@rotation_matrix_in_parent
    print(res)

    # static_transformStamped_5 = TransformStamped()
    # # Define the second static transform from 'hdr_base' to 'hdr_cam'
    # static_transformStamped_5.header.stamp = rospy.Time.now()
    # static_transformStamped_5.header.frame_id = "base"
    # static_transformStamped_5.child_frame_id = "wide_angle_camera_rear_camera"
    # static_transformStamped_5.transform.translation.x = -0.4087
    # static_transformStamped_5.transform.translation.y = 0.0
    # static_transformStamped_5.transform.translation.z = 0.0205

    # # Convert Euler angles to a quaternion
    # x, y, z = -0.4087, 0.0, 0.0205
    # translation_matrix = tf.transformations.translation_matrix((x, y, z))
    # roll, pitch, yaw = 0.0, 0.0, 3.141592653589793
    # quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    # rotation_matrix_in_parent_wf = tf.transformations.quaternion_matrix(quaternion)
    # rotation_matrix_in_parent_wf=translation_matrix@rotation_matrix_in_parent_wf
    # static_transformStamped_5.transform.rotation.x = quaternion[0]
    # static_transformStamped_5.transform.rotation.y = quaternion[1]
    # static_transformStamped_5.transform.rotation.z = quaternion[2]
    # static_transformStamped_5.transform.rotation.w = quaternion[3]
    
    # static_transformStamped_6 = TransformStamped()
    # # Define the second static transform from 'hdr_base' to 'hdr_cam'
    # static_transformStamped_6.header.stamp = rospy.Time.now()
    # static_transformStamped_6.header.frame_id = "wide_angle_camera_rear_camera"
    # static_transformStamped_6.child_frame_id = "wide_angle_camera_rear_camera_parent"
    # static_transformStamped_6.transform.translation.x = -0.00421
    # static_transformStamped_6.transform.translation.y = 0.0
    # static_transformStamped_6.transform.translation.z = 0.0

    # # Convert Euler angles to a quaternion
    # x,y,z=-0.00421,0.0,0.0
    # translation_matrix = tf.transformations.translation_matrix((x, y, z))
    # roll, pitch, yaw = -1.5707963267948966, 0.0, -1.5707963267948966
    # quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    # rotation_matrix_in_parent_fo = tf.transformations.quaternion_matrix(quaternion)
    # rotation_matrix_in_parent_fo=translation_matrix@rotation_matrix_in_parent_fo
    # static_transformStamped_6.transform.rotation.x = quaternion[0]
    # static_transformStamped_6.transform.rotation.y = quaternion[1]
    # static_transformStamped_6.transform.rotation.z = quaternion[2]
    # static_transformStamped_6.transform.rotation.w = quaternion[3]
    # ok=rotation_matrix_in_parent_wf@rotation_matrix_in_parent_fo
    # print(ok)
    
    # transformation_matrix =np.array([[ 1.22464680e-16, -1.11022302e-16, -1.00000000e+00,  -4.04490000e-01],
    #                                         [ 1.00000000e+00,  2.22044605e-16,  1.14423775e-17,  -5.15576302e-19],
    #                                         [ 0.00000000e+00 ,-1.00000000e+00,  0.00000000e+00 , 2.05000000e-02],
    #                                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
    # static_transformStamped_7 = TransformStamped()

    # # Extract translation from the transformation matrix
    # static_transformStamped_7.transform.translation.x = transformation_matrix[0, 3]
    # static_transformStamped_7.transform.translation.y = transformation_matrix[1, 3]
    # static_transformStamped_7.transform.translation.z = transformation_matrix[2, 3]

    # # Convert the rotation matrix to a quaternion
    # # rotation_matrix = transformation_matrix[:3, :3]
    # quaternion = tf.transformations.quaternion_from_matrix(transformation_matrix)

    # # Set the rotation in the message
    # static_transformStamped_7.transform.rotation.x = quaternion[0]
    # static_transformStamped_7.transform.rotation.y = quaternion[1]
    # static_transformStamped_7.transform.rotation.z = quaternion[2]
    # static_transformStamped_7.transform.rotation.w = quaternion[3]

    # # Set the header and child_frame_id as needed
    # static_transformStamped_7.header.stamp = rospy.Time.now()
    # static_transformStamped_7.header.frame_id = "base"
    # static_transformStamped_7.child_frame_id = "check"

    static_transformStamped_5 = TransformStamped()
    # Define the second static transform from 'hdr_base' to 'hdr_cam'
    static_transformStamped_5.header.stamp = rospy.Time.now()
    static_transformStamped_5.header.frame_id = "base"
    static_transformStamped_5.child_frame_id = "wide_angle_camera_front_camera"
    static_transformStamped_5.transform.translation.x = 0.4087
    static_transformStamped_5.transform.translation.y = 0.0
    static_transformStamped_5.transform.translation.z = 0.0205

    # Convert Euler angles to a quaternion
    x, y, z = 0.4087, 0.0, 0.0205
    translation_matrix = tf.transformations.translation_matrix((x, y, z))
    roll, pitch, yaw = 0.0, 0.0, -0.0
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    rotation_matrix_in_parent_wf = tf.transformations.quaternion_matrix(quaternion)
    rotation_matrix_in_parent_wf=translation_matrix@rotation_matrix_in_parent_wf
    static_transformStamped_5.transform.rotation.x = quaternion[0]
    static_transformStamped_5.transform.rotation.y = quaternion[1]
    static_transformStamped_5.transform.rotation.z = quaternion[2]
    static_transformStamped_5.transform.rotation.w = quaternion[3]
    
    static_transformStamped_6 = TransformStamped()
    # Define the second static transform from 'hdr_base' to 'hdr_cam'
    static_transformStamped_6.header.stamp = rospy.Time.now()
    static_transformStamped_6.header.frame_id = "wide_angle_camera_front_camera"
    static_transformStamped_6.child_frame_id = "wide_angle_camera_front_camera_parent"
    static_transformStamped_6.transform.translation.x = -0.00421
    static_transformStamped_6.transform.translation.y = 0.0
    static_transformStamped_6.transform.translation.z = 0.0

    # Convert Euler angles to a quaternion
    x,y,z=-0.00421,0.0,0.0
    translation_matrix = tf.transformations.translation_matrix((x, y, z))
    roll, pitch, yaw = -1.5707963267948966, 0.0, -1.5707963267948966
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    rotation_matrix_in_parent_fo = tf.transformations.quaternion_matrix(quaternion)
    rotation_matrix_in_parent_fo=translation_matrix@rotation_matrix_in_parent_fo
    static_transformStamped_6.transform.rotation.x = quaternion[0]
    static_transformStamped_6.transform.rotation.y = quaternion[1]
    static_transformStamped_6.transform.rotation.z = quaternion[2]
    static_transformStamped_6.transform.rotation.w = quaternion[3]
    ok=rotation_matrix_in_parent_wf@rotation_matrix_in_parent_fo
    print(ok)
    
    transformation_matrix =np.array([[-3.63509055e-06 , 1.43680318e-01 , 9.89624154e-01  ,3.53700000e-01],
                                            [ 1.00000000e+00, -1.34923184e-11 , 3.67320510e-06  ,0.00000000e+00],
                                            [ 5.27780629e-07 , 9.89624154e-01 ,-1.43680318e-01 , 1.63400000e-01],
                                            [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    static_transformStamped_7 = TransformStamped()

    # Extract translation from the transformation matrix
    static_transformStamped_7.transform.translation.x = transformation_matrix[0, 3]
    static_transformStamped_7.transform.translation.y = transformation_matrix[1, 3]
    static_transformStamped_7.transform.translation.z = transformation_matrix[2, 3]

    # Convert the rotation matrix to a quaternion
    # rotation_matrix = transformation_matrix[:3, :3]
    quaternion = tf.transformations.quaternion_from_matrix(transformation_matrix)

    # Set the rotation in the message
    static_transformStamped_7.transform.rotation.x = quaternion[0]
    static_transformStamped_7.transform.rotation.y = quaternion[1]
    static_transformStamped_7.transform.rotation.z = quaternion[2]
    static_transformStamped_7.transform.rotation.w = quaternion[3]

    # Set the header and child_frame_id as needed
    static_transformStamped_7.header.stamp = rospy.Time.now()
    static_transformStamped_7.header.frame_id = "base"
    static_transformStamped_7.child_frame_id = "check"
    
    static_transformStamped_8 = TransformStamped()
    # Define the second static transform from 'hdr_base' to 'hdr_cam'
    static_transformStamped_8.header.stamp = rospy.Time.now()
    static_transformStamped_8.header.frame_id = "base"
    static_transformStamped_8.child_frame_id = "lidar_parent"
    static_transformStamped_8.transform.translation.x = -0.310
    static_transformStamped_8.transform.translation.y = 0.0
    static_transformStamped_8.transform.translation.z = 0.1585

    # Convert Euler angles to a quaternion
    x,y,z=-0.310,0.0,0.1585
    translation_matrix = tf.transformations.translation_matrix((x, y, z))
    roll, pitch, yaw = 0, 0.0, 1.5707963267948966
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    rotation_matrix_in_parent_fo = tf.transformations.quaternion_matrix(quaternion)
    rotation_matrix_in_parent_fo=translation_matrix@rotation_matrix_in_parent_fo
    static_transformStamped_8.transform.rotation.x = quaternion[0]
    static_transformStamped_8.transform.rotation.y = quaternion[1]
    static_transformStamped_8.transform.rotation.z = quaternion[2]
    static_transformStamped_8.transform.rotation.w = quaternion[3]
    ok=rotation_matrix_in_parent_wf@rotation_matrix_in_parent_fo
    print("lidar_parents:",ok)
    # print("if reversed:",rotation_matrix_in_parent_wf@(rotation_matrix_in_parent_fo@translation_matrix))
    
    transformation_matrix =np.array([[-3.63509055e-06 , 1.43680318e-01 , 9.89624154e-01  ,3.53700000e-01],
                                            [ 1.00000000e+00, -1.34923184e-11 , 3.67320510e-06  ,0.00000000e+00],
                                            [ 5.27780629e-07 , 9.89624154e-01 ,-1.43680318e-01 , 1.63400000e-01],
                                            [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    static_transformStamped_9 = TransformStamped()

    # Extract translation from the transformation matrix
    static_transformStamped_9.transform.translation.x = transformation_matrix[0, 3]
    static_transformStamped_9.transform.translation.y = transformation_matrix[1, 3]
    static_transformStamped_9.transform.translation.z = transformation_matrix[2, 3]

    # Convert the rotation matrix to a quaternion
    # rotation_matrix = transformation_matrix[:3, :3]
    quaternion = tf.transformations.quaternion_from_matrix(transformation_matrix)

    # Set the rotation in the message
    static_transformStamped_9.transform.rotation.x = quaternion[0]
    static_transformStamped_9.transform.rotation.y = quaternion[1]
    static_transformStamped_9.transform.rotation.z = quaternion[2]
    static_transformStamped_9.transform.rotation.w = quaternion[3]

    # Set the header and child_frame_id as needed
    static_transformStamped_9.header.stamp = rospy.Time.now()
    static_transformStamped_9.header.frame_id = "base"
    static_transformStamped_9.child_frame_id = "check"
    
    
    
    # Broadcast the second transform
    broadcaster.sendTransform([static_transformStamped, static_transformStamped_2, static_transformStamped_3, static_transformStamped_4, static_transformStamped_5, static_transformStamped_6, static_transformStamped_7])
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
