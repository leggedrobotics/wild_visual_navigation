from liegroups.torch import SE3, SO3
from torch_geometric.data import Data
import os
import torch
import torch.nn.functional as F
from typing import Optional



class BaseNode:
    """Base node data structure"""

    _name = "base_node"

    def __init__(self, timestamp: float = 0.0, pose_base_in_world: torch.tensor = torch.eye(4)):
        assert isinstance(pose_base_in_world, torch.Tensor)

        self._timestamp = timestamp
        self._pose_base_in_world = pose_base_in_world

    def __str__(self):
        return f"{self._name}_{self._timestamp}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self._name == other.name
            and self._timestamp == other.timestamp
            and torch.equal(self._pose_base_in_world, other.pose_base_in_world)
        )

    def __lt__(self, other):
        return self._timestamp < other.timestamp

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._pose_base_in_world = self._pose_base_in_world.to(device)
        
    @classmethod
    def from_node(cls, instance):
        return cls(timestamp=instance.timestamp, pose_base_in_world=instance.pose_base_in_world)

    def is_valid(self):
        return True

    def pose_between(self, other):
        """Computes pose difference (SE(3)) between this state and other

        Args:
            other (BaseNode): Other state

        Returns:
            tensor (torch.tensor): Pose difference expressed in this' frame
        """
        return other.pose_base_in_world.inverse() @ self.pose_base_in_world

    def distance_to(self, other):
        """Computes the relative distance between states

        Args:
            other (BaseNode): Other state

        Returns:
            distance (float): absolute distance between the states
        """
        # Compute pose difference, then log() to get a vector, then extract position coordinates, finally get norm
        return (
            SE3.from_matrix(self.pose_base_in_world.inverse() @ other.pose_base_in_world, normalize=True)
            .log()[:3]
            .norm()
        )

    @property
    def name(self):
        return self._name
    
    @property
    def pose_base_in_world(self):
        return self._pose_base_in_world

    @property
    def timestamp(self):
        return self._timestamp

    @pose_base_in_world.setter
    def pose_base_in_world(self, pose_base_in_world: torch.tensor):
        self._pose_base_in_world = pose_base_in_world

    @timestamp.setter
    def timestamp(self, timestamp: float):
        self._timestamp = timestamp