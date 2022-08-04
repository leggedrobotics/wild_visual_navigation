from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.utils import make_box, make_plane
from liegroups.torch import SE3, SO3
from torch_geometric.data import Data
import os
import torch


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
        return SE3.from_matrix(self.pose_base_in_world.inverse() @ other.pose_base_in_world).log()[:3].norm()

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


class MissionNode(BaseNode):
    """Mission node stores the minimum information required for traversability estimation
    All the information is stored on the image plane"""

    _name = "mission_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_cam_in_base: torch.tensor = torch.eye(4),
        pose_cam_in_world: torch.tensor = None,
        image: torch.tensor = None,
        image_projector: ImageProjector = None,
    ):
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)
        # Initialize members
        self._pose_cam_in_base = pose_cam_in_base
        self._pose_cam_in_world = (
            self._pose_base_in_world @ self._pose_cam_in_base if pose_cam_in_world is None else pose_cam_in_world
        )
        self._image = image
        self._image_projector = image_projector

        # Uninitialized members
        self._features = None
        self._feature_type = None
        self._feature_edges = None
        self._feature_segments = None
        self._feature_positions = None
        self._prediction = None
        self._supervision_mask = None
        self._supervision_signal = None
        self._supervision_signal_valid = None

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._image_projector.change_device(device)
        self._image = self._image.to(device)
        self._pose_cam_in_base = self._pose_cam_in_base.to(device)
        self._pose_cam_in_world = self._pose_cam_in_world.to(device)

        if self._features is not None:
            self._features = self._features.to(device)
        if self._feature_edges is not None:
            self._feature_edges = self._feature_edges.to(device)
        if self._feature_segments is not None:
            self._feature_segments = self._feature_segments.to(device)
        if self._feature_positions is not None:
            self._feature_positions = self._feature_positions.to(device)
        if self._prediction is not None:
            self._prediction = self._prediction.to(device)
        if self._supervision_mask is not None:
            self._supervision_mask = self._supervision_mask.to(device)
        if self._supervision_signal is not None:
            self._supervision_signal = self._supervision_signal.to(device)
        if self._supervision_signal_valid is not None:
            self._supervision_signal_valid = self._supervision_signal_valid.to(device)

    def as_pyg_data(self):
        return Data(
            x=self.features,
            edge_index=self._feature_edges,
            y=self._supervision_signal,
            y_valid=self._supervision_signal_valid,
        )

    def is_valid(self):
        return isinstance(self._features, torch.Tensor) and isinstance(self._supervision_signal, torch.Tensor)

    @property
    def features(self):
        return self._features

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def feature_edges(self):
        return self._feature_edges

    @property
    def feature_segments(self):
        return self._feature_segments

    @property
    def feature_positions(self):
        return self._feature_positions

    @property
    def image(self):
        return self._image

    @property
    def image_projector(self):
        return self._image_projector

    @property
    def pose_cam_in_world(self):
        return self._pose_cam_in_world

    @property
    def prediction(self):
        return self.prediction

    @property
    def supervision_signal(self):
        return self._supervision_signal

    @property
    def supervision_mask(self):
        return self._supervision_mask

    @features.setter
    def features(self, features):
        self._features = features

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type

    @feature_edges.setter
    def feature_edges(self, feature_edges):
        self._feature_edges = feature_edges

    @feature_segments.setter
    def feature_segments(self, feature_segments):
        self._feature_segments = feature_segments

    @feature_positions.setter
    def feature_positions(self, feature_positions):
        self._feature_positions = feature_positions

    @image.setter
    def image(self, image):
        self._image = image

    @image_projector.setter
    def image_projector(self, image_projector):
        self._image_projector = image_projector

    @pose_cam_in_world.setter
    def pose_cam_in_world(self, pose_cam_in_world):
        self._pose_cam_in_world = pose_cam_in_world

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = prediction

    @supervision_signal.setter
    def supervision_signal(self, _supervision_signal):
        self._supervision_signal = _supervision_signal

    @supervision_mask.setter
    def supervision_mask(self, supervision_mask):
        self._supervision_mask = supervision_mask

    def save(self, output_path: str, index: int, graph_only: bool = False):
        if self._feature_positions is not None:
            graph_data = self.as_pyg_data()
            path = os.path.join(output_path, "graph", f"graph_{index:06d}.pt")
            torch.save(graph_data, path)
            if not graph_only:
                p = path.replace("graph", "img")
                torch.save(self._image.cpu(), p)

                p = path.replace("graph", "center")
                torch.save(self._feature_positions.cpu(), p)

                p = path.replace("graph", "seg")
                torch.save(self._feature_segments.cpu(), p)

    def update_supervision_signal(self):
        if self._supervision_mask is None:
            return

        if len(self._supervision_mask.shape) == 3:
            signal = self._supervision_mask.nanmean(axis=0)

        # If we don't have features, return
        if self._features is None:
            return

        # If we have features, update supervision signal
        labels_per_segment = []
        for s in range(self._feature_segments.max() + 1):
            # Get a mask indices for the segment
            m = self.feature_segments == s
            # Add the higehst number per segment
            # labels_per_segment.append(signal[m].max())
            labels_per_segment.append(signal[m].nanmean(axis=0))

        # Prepare supervision signal
        torch_labels = torch.stack(labels_per_segment)
        # if torch_labels.sum() > 0:
        self._supervision_signal = torch.nan_to_num(torch_labels, nan=0)
        # Binary mask
        self._supervision_signal_valid = torch_labels > 0


class ProprioceptionNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    _name = "local_proprioception_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_footprint_in_base: torch.tensor = torch.eye(4),
        pose_footprint_in_world: torch.tensor = None,
        length: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
        proprioception: torch.tensor = None,
        traversability: torch.tensor = torch.FloatTensor([0.0]),
        traversability_var: torch.tensor = torch.FloatTensor([0.0]),
    ):
        assert isinstance(pose_base_in_world, torch.Tensor)
        assert isinstance(pose_footprint_in_base, torch.Tensor)
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)

        self._pose_footprint_in_base = pose_footprint_in_base
        self._pose_footprint_in_world = (
            self._pose_base_in_world @ self._pose_footprint_in_base
            if pose_footprint_in_world is None
            else pose_footprint_in_world
        )
        self._length = length
        self._width = width
        self._height = height
        self._proprioceptive_state = proprioception
        self._traversability = traversability
        self._traversability_var = traversability_var

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._pose_footprint_in_base = self._pose_footprint_in_base.to(device)
        self._pose_footprint_in_world = self._pose_footprint_in_world.to(device)
        self._proprioceptive_state = self._proprioceptive_state.to(device)

    def get_bounding_box_points(self):
        return make_box(self._length, self._width, self._height, pose=self._pose_base_in_world, grid_size=5).to(
            self._pose_base_in_world.device
        )

    def get_footprint_points(self):
        return make_plane(x=self._length, y=self._width, pose=self._pose_footprint_in_world, grid_size=25).to(
            self._pose_footprint_in_world.device
        )

    def get_side_points(self):
        return make_plane(x=0.0, y=self._width, pose=self._pose_footprint_in_world, grid_size=2).to(
            self._pose_footprint_in_world.device
        )

    def update_traversability(self, traversability: torch.tensor, traversability_var: torch.tensor):
        self._traversability_var = 1.0 / (1.0 / self._traversability_var ** 2 + 1.0 / traversability_var ** 2)
        self._traversability = self.traversability_var * (
            1.0 / self._traversability_var * self._traversability + 1.0 / traversability_var * traversability
        )

    @property
    def traversability(self):
        return self._traversability

    @property
    def traversability_var(self):
        return self._traversability_var

    @property
    def pose_footprint_in_world(self):
        return self._pose_footprint_in_world

    @property
    def propropioceptive_state(self):
        return self._proprioceptive_state

    @traversability.setter
    def traversability(self, traversability):
        self._traversability = traversability

    @traversability_var.setter
    def traversability_var(self, variance):
        self._traversability_var = variance

    def is_valid(self):
        return isinstance(self._proprioceptive_state, torch.Tensor)


def run_base_state():
    """TODO."""

    import torch

    rs1 = BaseNode(1, pose_base_in_world=SE3(SO3.identity(), torch.Tensor([1, 0, 0])).as_matrix())
    rs2 = BaseNode(2, pose_base_in_world=SE3(SO3.identity(), torch.Tensor([2, 0, 0])).as_matrix())

    # Check that distance between robot states is correct
    assert abs(rs2.distance_to(rs1) - 1.0) < 1e-10

    # Check that objects are different
    assert rs1 != rs2

    # Check that timestamps are 1 second apart
    assert rs2.timestamp - rs1.timestamp == 1.0

    # Create node from another one
    rs3 = BaseNode.from_node(rs1)
    assert rs3 == rs1


if __name__ == "__main__":
    run_base_state()
