import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class FootFilter:
    def __init__(self, name="RF_FOOT"):
        self.name = name
        self.poses = []
        self.filtered_poses = []
        self.max_poses_window = 200
        self.steps_to_compare=1
        self.min_contact_duration=0.2
        self.last_contact_time=None
        self.last_contact_position = None
        self.max_xy_distance=0.1
        self.continuous_contact_count = 0
        self.contact_threshold = 8 
    
    def filter_to_single(self, current_pose, estimated_contact: Optional[bool] = None, current_time: float = None):
        """
        current_pose: [x, y, z, rx, ry, rz, rw]
        Filters the foot contact state from a noisy state estimator using a temporal method.
        It'll preserve only one contact state in a sequence of contact states.
        Not useful in my case.
        """
 
        # Add to the buffer
        current_pose = np.array(current_pose)
        if len(self.poses) > self.max_poses_window:
            self.poses = self.poses[-self.max_poses_window:]
            self.filtered_poses = self.filtered_poses[-self.max_poses_window:]
            plot_foot_data(self)

        filtered_contact = False
        if estimated_contact:
            self.continuous_contact_count += 1
            if self.continuous_contact_count >= self.contact_threshold:
                filtered_contact = True
        else:
            self.continuous_contact_count = 0
        # Check for the duration since last contact
        if estimated_contact and self.last_contact_position is not None:
            time_since_last_contact = current_time - self.last_contact_time
            is_short_duration = time_since_last_contact < self.min_contact_duration
            xy_distance = np.linalg.norm(current_pose[:2] - self.last_contact_position[:2])

           # Check rising motion in the last few steps
            if len(self.poses) >= self.steps_to_compare:
                recent_z_positions = [pose[0][2] for pose in self.poses[-self.steps_to_compare:]]
                if all(current_pose[2] > z for z in recent_z_positions):
                    self.poses.append((current_pose, estimated_contact, current_time))
                    self.filtered_poses.append((current_pose, False, current_time))
                    return False  # Foot is consistently rising, unlikely to be in contact

            if is_short_duration or xy_distance < self.max_xy_distance:
                self.poses.append((current_pose, estimated_contact, current_time))
                self.filtered_poses.append((current_pose, False, current_time))
                return False 
  
        # Update last contact details if current state is a contact
        if filtered_contact:
            self.last_contact_time = current_time
            self.last_contact_position = current_pose
        self.filtered_poses.append((current_pose, filtered_contact, current_time))
        self.poses.append((current_pose, estimated_contact, current_time))

            
        return filtered_contact

    def filter(self, current_pose, estimated_contact: Optional[bool] = None, current_time: float = None):
        """
        current_pose: [x, y, z, rx, ry, rz, rw]
        Filters the foot contact state from a noisy state estimator using a temporal method.
        This one will preserve most contact states in a sequence of contact states.
        """

        # Add to the buffer
        current_pose = np.array(current_pose)
        if len(self.poses) > self.max_poses_window:
            self.poses = self.poses[-self.max_poses_window:]
            self.filtered_poses = self.filtered_poses[-self.max_poses_window:]
            # plot_foot_data(self)

        # Sliding window for the contact states
        window_size = 5
        contact_threshold = 3  # Minimum number of contacts in the window to consider as valid contact
        contact_window = [pose[1] for pose in self.poses[-window_size:]]  # Get the last 10 contact states
        # Check if the number of contacts in the window meets or exceeds the threshold
        if contact_window.count(True) >= contact_threshold and estimated_contact:
            filtered_contact = True
        else:
            filtered_contact = False

        # Update last contact details if current state is a contact
        if filtered_contact:
            self.last_contact_time = current_time
            self.last_contact_position = current_pose
        self.filtered_poses.append((current_pose, filtered_contact, current_time))
        self.poses.append((current_pose, estimated_contact, current_time))

            
        return filtered_contact

def plot_foot_data(foot_filter:FootFilter):
    poses=foot_filter.filtered_poses
    # poses=foot_filter.poses
    z_values = [pose[0][2] for pose in poses]
    contact_states = [pose[1] for pose in poses]

    plt.figure(figsize=(10, 6))
    plt.plot(z_values, label=f'Z Position of {foot_filter.name}')
    
    # Annotate contact poses
    for i, (z, contact) in enumerate(zip(z_values, contact_states)):
        if contact:
            plt.scatter(i, z, color='red')  # Annotate contact points in red

    plt.xlabel('Pose Index')
    plt.ylabel('Z Position')
    plt.title('Foot Filter Data Analysis')
    plt.legend()
    # plt.show()
    plt.savefig(f"foot_filter_{foot_filter.name}.png")