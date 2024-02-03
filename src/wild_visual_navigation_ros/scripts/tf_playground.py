import rospy
import tf
import tf2_ros
import numpy as np
import ros_converter as rc
from geometry_msgs.msg import TransformStamped
from msg_to_transmatrix import pq_to_se3
class StaticTransform():
    def __init__(self,translation,quatoreuler,parent,child) -> None:
        
        if len(quatoreuler)==4:
            quat=quatoreuler
        elif len(quatoreuler)==3:
            quat=tf.transformations.quaternion_from_euler(*quatoreuler)
        else:
            raise ValueError("quaternion or euler angles must be given")
        
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
        self.euler=tf.transformations.euler_from_quaternion(quat)
    @classmethod
    def create_from_matrix(cls, matrix: np.ndarray, parent: str, child: str) -> 'StaticTransform':
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        translation = matrix[:3, 3]
        quat = tf.transformations.quaternion_from_matrix(matrix)
        return cls(translation, quat, parent, child)
    
    @staticmethod
    def multiply(A:'StaticTransform',B:'StaticTransform') -> 'StaticTransform':
        """ 
        The order of A/B is T^{parent}_{child}, after (A*B) multiplication, the new parent will be A's parent
        while the new child will be B's child
        """
        new_matrix=A.transformation_matrix@B.transformation_matrix
        new_parent=A.parent
        new_child=B.child
        return StaticTransform.create_from_matrix(new_matrix,new_parent,new_child)
        

def publish_transform():
    rospy.init_node('tf_broadcaster')

    all_tf=[]
    # Create a single TF2 broadcaster
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Define the first static transform from 'base' to 'hdr_base'
    stf_1=StaticTransform([0.3537,0.0,0.1634],[0.997,0.0,-0.072,0.0],"base","hdr_base")
    all_tf.append(stf_1.static_transformStamped)
   
    stf_2=StaticTransform([0,0,0],tf.transformations.quaternion_from_euler(-1.5708, 0.0, -1.5708),"hdr_base","hdr_cam")
    all_tf.append(stf_2.static_transformStamped)
    
    # rotate base frame around z axis by 180 degree, pointing back
    T_0=np.array([[-1,0,0,0],
                  [0,-1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    
    stf3_1=StaticTransform.create_from_matrix(T_0,'base','mirrored_base')
    # all_tf.append(stf3_1.static_transformStamped)
    
    suc, T_1 = rc.ros_tf_to_numpy(((0.3537,0.0,0.1634), (0.997,0.0,-0.072,0.0)))
    stf3_2=StaticTransform.create_from_matrix(T_1,'mirrored_base','hdr_rear_base')
    # all_tf.append(stf3_2.static_transformStamped)
    
    base_to_hdr_rear_base=StaticTransform.multiply(stf3_1,stf3_2)
    all_tf.append(base_to_hdr_rear_base.static_transformStamped)
    

    stf4=StaticTransform([0,0,0],tf.transformations.quaternion_from_euler(-1.5708, 0.0, -1.5708),"hdr_rear_base","hdr_rear_cam")
    all_tf.append(stf4.static_transformStamped)
    
    stf5=StaticTransform([0.4087,0.0,0.0205],tf.transformations.quaternion_from_euler(0.0, 0.0, -0.0),"base","wide_angle_camera_front_camera")
    all_tf.append(stf5.static_transformStamped)

    stf6=StaticTransform([-0.00421,0,0],tf.transformations.quaternion_from_euler(-1.5708, 0.0, -1.5708),"wide_angle_camera_front_camera","wide_angle_camera_front_camera_parent")
    all_tf.append(stf6.static_transformStamped)
    

    
    stf8=StaticTransform([-0.310,0.0,0.1585],(0.0, 0.0, 1.5707963267948966),"base","lidar_parent")
    # all_tf.append(stf8.static_transformStamped)
    
    stf8_1=StaticTransform([0,0,0],(0.0, 0.0, 1.5707963267948966),'lidar_parent','lidar')
    # all_tf.append(stf8_1.static_transformStamped)
    
    base_to_lidar=StaticTransform.multiply(stf8,stf8_1)
    all_tf.append(base_to_lidar.static_transformStamped)

    
    # Broadcast the second transform
    broadcaster.sendTransform(all_tf)
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
