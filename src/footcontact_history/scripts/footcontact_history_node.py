import rospy
import math
import tf2_ros
from anymal_msgs.msg import AnymalState
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
import time
from matplotlib import pyplot as plt
from typing import Optional

def query_tf(parent_frame: str, child_frame: str, stamp: Optional[rospy.Time] = None, from_message: Optional[AnymalState] = None):
    """Helper function to query TFs

    Args:
        parent_frame (str): Frame of the parent TF
        child_frame (str): Frame of the child
        from_message (AnymalState, optional): AnymalState message containing the TFs
    """
    if from_message is None:
        # error, must have from_message
        raise ValueError("Must provide from_message")
    else:
        
        for foot_transform in from_message.contacts:
            if foot_transform.name == child_frame and foot_transform.header.frame_id==parent_frame:
                trans=(foot_transform.position.x,
                        foot_transform.position.y,
                        foot_transform.position.z)
                # FOOT is merely a point, no rotation
                rot=np.array([0,0,0,1])
                return (trans,tuple(rot))

        # If the requested TF is not found in the message
        # rospy.logwarn(f"Couldn't find tf between {parent_frame} and {child_frame} in the provided message")
        raise ValueError(f"Couldn't find tf between {parent_frame} and {child_frame} in the provided message")

class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        
        x = np.linspace(0, 99, 100)
        y = np.sin(x)

        (self.line_heights,) = self.ax.plot(
            x, y, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green"
        )
        (self.line_contacts,) = self.ax.plot(
            x, y, marker="o", markersize=20, markeredgecolor="blue", markerfacecolor="green"
        )

    def plot(self, ana, keep):
        x = np.arange(len(ana))
        heights = ana[:, 2]
        self.line_heights.set_ydata(heights)
        self.line_heights.set_xdata(x)

        self.line_contacts.set_ydata(heights[keep])
        self.line_contacts.set_xdata(x[keep])

        self.ax.autoscale(enable=True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class FootTracer:
    def __init__(self, name="RF_FOOT", visu=False):
        self.visu = visu
        self.name = name

        self.poses = []
        self.footholds = []
        # self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)
        # subscribe to anymal_state and get the transform from there
        self.publish = rospy.get_param("/footcontact_history/publish",True)
        self.visu = rospy.get_param("/footcontact_history/visu",False)
        self.reference_frame = rospy.get_param("/footcontact_history/reference_frame","odom")

        self.max_poses_window = rospy.get_param("/footcontact_history/max_poses_window",100)
        self.nr_of_contacts_per_feet = rospy.get_param("/footcontact_history/nr_of_contacts_per_feet",100)
        self.evaluate_every = rospy.get_param("/footcontact_history/evaluate_every",10)
        self.air_time_foot = rospy.get_param("/footcontact_history/air_time_foot",20)
        self.local_minima_window = rospy.get_param("/footcontact_history/local_minima_window",3)
        self.border_reject = rospy.get_param("/footcontact_history/border_reject",20)
        self.distance_threshold_buffer = rospy.get_param("/footcontact_history/distance_threshold_buffer",0.05)

        self.step = 0
        self.state_sub = rospy.Subscriber("/state_estimator/anymal_state", AnymalState, self.state_callback)

        if self.publish:
            self.pub = rospy.Publisher("footcontact_history/" + name, PointCloud2, queue_size=10)

        if self.visu:
            self.visualizer = Visualizer()

    def state_callback(self,msg:AnymalState):
    
        # trans = self.tfBuffer.lookup_transform(self.reference_frame, self.name, rospy.Time())
        pose_foot_in_world =query_tf(self.reference_frame, self.name, from_message=msg)
            
        t = pose_foot_in_world[0]
        r = pose_foot_in_world[1]
        # pose = np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w])
        pose=np.array(t+r)

        self.poses.append(pose)
        if len(self.poses) > self.max_poses_window:
            self.poses = self.poses[-self.max_poses_window :]

        if self.step % self.evaluate_every == 0 and len(self.poses) == self.max_poses_window:
            self.step = 0
            self.eval_foot_trajectory()

        self.step += 1

    def eval_foot_trajectory(self):
        foot_positions = np.stack(self.poses)

        ana = foot_positions

        mi = ana.min()
        ma = ana.max()

        k = 20
        idx = np.argpartition(-ana[:, 2], -k)[-k:]
        candidate_indices = idx[np.argsort(-ana[idx, 2])][::-1]  # Indices sorted by value from largest to smallest

        keep = [candidate_indices[0]]

        # Filter temporal
        for c in candidate_indices[1:]:
            if (np.abs(np.array(keep) - c)).min() > self.air_time_foot:
                keep.append(c)

        # Refine to lowes in small area
        for i, k in enumerate(keep):
            mi = max(0, k - self.local_minima_window)
            ma = min(len(ana) - 1, k + self.local_minima_window)
            keep[i] = np.argmin(ana[mi:ma, 2]) + mi

        # Remove local minima at border of window
        keep = [k for k in keep if k > self.border_reject and k < len(self.poses) - self.border_reject]

        # Check against foothold history before adding
        if len(self.footholds) > 0:
            for k in keep:
                if np.min(np.linalg.norm(np.array(self.footholds) - ana[k, :3], axis=1)) > self.distance_threshold_buffer:
                    self.footholds.append(ana[k, :3])
        else:
            for k in keep:
                self.footholds.append(ana[k, :3])

        if len(self.footholds) > self.nr_of_contacts_per_feet:
            self.footholds = self.footholds[-self.nr_of_contacts_per_feet :]

        self.publish_footholds()

        if self.visu:
            self.visualizer.plot(ana, keep)

    def publish_footholds(self):
        if len(self.footholds) == 0:
            return

        da = {}
        da = np.zeros(
            len(self.footholds),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
            ],
        )
        np_footholds = np.array(self.footholds)
        da["x"] = np_footholds[:, 0]
        da["y"] = np_footholds[:, 1]
        da["z"] = np_footholds[:, 2]

        msg = ros_numpy.msgify(PointCloud2, da)
        msg.header.frame_id = self.reference_frame
        msg.header.seq = 0
        t = rospy.Time.now()
        msg.header.stamp = rospy.Time.now()
        self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("footcontact_history")

    feet = rospy.get_param("/footcontact_history/feet_frames",["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    print("Started footcontact_history node!")
    print("Tracking the feet: ", feet)
    fts = [FootTracer(n) for n in feet]
    rospy.spin()
    # rate = rospy.Rate(50)
    # ma = 0
    # while not rospy.is_shutdown():
    #     for foot in fts:
    #         pass
    #     rate.sleep()
    print("Exited footcontact_history node!")
