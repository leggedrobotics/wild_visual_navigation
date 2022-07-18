from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.utils import make_box, make_rounded_box, make_plane
from liegroups.torch import SE3, SO3
from PIL import Image, ImageDraw
from torch_geometric.data import Data
import kornia
import numpy as np
import os
import torch


class BaseNode:
    """Base node data structure"""

    name = "base_node"

    def __init__(self, timestamp: float = 0.0, T_WB: torch.tensor = torch.eye(4)):
        assert isinstance(T_WB, torch.Tensor)

        self.timestamp = timestamp
        self.T_WB = T_WB

    def __str__(self):
        return f"{self.name}_{self.timestamp}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if other is None:
            return False
        return self.name == other.name and self.timestamp == other.timestamp and torch.equal(self.T_WB, other.T_WB)

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    @classmethod
    def from_node(cls, instance):
        return cls(timestamp=instance.get_timestamp(), T_WB=instance.get_pose_base_in_world())

    def is_valid(self):
        return True

    def pose_between(self, other):
        """Computes pose difference (SE(3)) between this state and other

        Args:
            other (BaseNode): Other state

        Returns:
            tensor (torch.tensor): Pose difference expressed in this' frame
        """
        return other.T_WB.inverse() @ self.T_WB

    def distance_to(self, other):
        """Computes the relative distance between states

        Args:
            other (BaseNode): Other state

        Returns:
            distance (float): absolute distance between the states
        """
        # Compute pose difference, then log() to get a vector, then extract position coordinates, finally get norm
        return SE3.from_matrix(self.T_WB.inverse() @ other.T_WB).log()[:3].norm()

    def get_pose_base_in_world(self):
        return self.T_WB

    def get_timestamp(self):
        return self.timestamp

    def set_pose_base_in_world(self, T_WB: torch.tensor):
        self.T_WB = T_WB

    def set_timestamp(self, timestamp: float):
        self.timestamp = timestamp


class GlobalNode(BaseNode):
    """Global node stores the minimum information required for traversability estimation
    All the information is stored on the image plane"""

    name = "global_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        T_WB: torch.tensor = torch.eye(4),
        features: torch.tensor = None,
        feature_type: torch.tensor = None,
        feature_edges: torch.tensor = None,
        feature_segments: torch.tensor = None,
        feature_positions: torch.tensor = None,
        supervision_signal: torch.tensor = None,
    ):
        super().__init__(timestamp=timestamp, T_WB=T_WB)
        self.features = features
        self.feature_type = feature_type
        self.feature_edges = feature_edges
        self.feature_segments = feature_segments
        self.feature_positions = feature_positions
        self.supervision_signal = supervision_signal

    def get_features(self):
        return self.features

    def get_feature_type(self):
        return self.feature_type

    def get_feature_edges(self):
        return self.feature_edges

    def get_feature_segments(self):
        return self.feature_segments

    def get_feature_positions(self):
        return self.feature_positions

    def get_supervision_signal(self):
        return self.supervision_signal

    def save(self, output_path: str, index: int):
        graph_data = self.as_pyg_data()
        path = os.path.join(output_path, f"graph_{index:06d}.pt")
        torch.save(graph_data, path)

    def set_features(
        self,
        feature_type: str,
        features: torch.tensor,
        edges: torch.tensor,
        segments: torch.tensor,
        positions: torch.tensor,
    ):
        self.feature_type = feature_type
        self.features = features
        self.feature_edges = edges
        self.feature_segments = segments
        self.feature_positions = positions

    def set_supervision_signal(self, signal: torch.tensor, is_image: bool = True):
        if not is_image:
            self.supervision_signal = signal
        else:
            if len(signal.shape) == 3:
                signal = signal.mean(axis=0)

            # Query feature positions and get labels
            labels_per_segment = []
            for s in range(self.feature_segments.max() + 1):
                # Get a mask indices for the segment
                m = (self.feature_segments == s)[0, 0]
                # Count the labels
                idx, counts = torch.unique(signal[m], return_counts=True)
                # append the labels
                labels_per_segment.append(idx[torch.argmax(counts)])

            # Prepare supervision signal
            self.supervision_signal = torch.stack(labels_per_segment)

    def is_valid(self):
        return isinstance(self.features, torch.Tensor) and isinstance(self.supervision_signal, torch.Tensor)

    def as_pyg_data(self):
        return Data(x=self.features, edge_index=self.feature_edges, y=self.supervision_signal)


class DebugNode(BaseNode):
    """Debug node stores images and full states for debugging"""

    name = "local_debug_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        T_WB: torch.tensor = torch.eye(4),
        traversability_mask: torch.tensor = None,
        labeled_image: torch.tensor = None,
    ):
        super().__init__(timestamp=timestamp, T_WB=T_WB)
        self.traversability_mask = None
        self.image = None
        self.node = None

    def get_image(self):
        return self.image

    def get_traversability_mask(self):
        return self.traversability_mask

    def get_training_image(self):
        if self.image is None or self.traversability_mask is None:
            return None
        img_np = kornia.utils.tensor_to_image(self.image)
        img_pil = Image.fromarray(np.uint8(img_np * 255))
        img_draw = ImageDraw.Draw(img_pil)

        trav_np = kornia.utils.tensor_to_image(self.traversability_mask)
        trav_pil = Image.fromarray(np.uint8(trav_np * 255))

        for i in range(self.node.feature_edges.shape[1]):
            a, b = self.node.feature_edges[0, i, 0], self.node.feature_edges[0, i, 1]
            line_params = self.node.feature_positions[0][a].tolist() + self.node.feature_positions[0][b].tolist()
            img_draw.line(line_params, fill=(255, 255, 255, 100), width=2)

        for i in range(self.node.feature_positions.shape[1]):
            params = self.node.feature_positions[0][i].tolist()
            color = trav_pil.getpixel((params[0], params[1]))
            r = 10
            params = [p - r for p in params] + [p + r for p in params]
            img_draw.ellipse(params, fill=color)

        np_draw = np.array(img_pil)
        return kornia.utils.image_to_tensor(np_draw.copy()).to(self.image.device)

    def set_image(self, image: torch.tensor):
        self.image = image

    def set_training_node(self, node):
        self.node = node

    def set_traversability_mask(self, mask: torch.tensor):
        self.traversability_mask = mask

    def is_valid(self):
        return isinstance(self.traversability_mask, torch.Tensor)


class LocalProprioceptionNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    name = "local_proprioception_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        T_WB: torch.tensor = torch.eye(4),
        T_BF: torch.tensor = torch.eye(4),
        T_WF: torch.tensor = None,
        length: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
        proprioception: torch.tensor = None,
    ):
        assert isinstance(T_WB, torch.Tensor)
        assert isinstance(T_BF, torch.Tensor)
        super().__init__(timestamp=timestamp, T_WB=T_WB)

        self.T_BF = T_BF
        # footprint in world
        self.T_WF = self.T_WB @ self.T_BF if T_WF is None else T_WF
        self.length = length
        self.width = width
        self.height = height
        self.proprioceptive_state = proprioception

    def get_bounding_box_points(self):
        return make_box(self.length, self.width, self.height, pose=self.T_WB, grid_size=5).to(self.T_WB.device)

    def get_footprint_points(self):
        return make_plane(x=self.length, y=self.width, pose=self.T_WF, grid_size=25).to(self.T_WF.device)

    def get_image(self):
        return self.image

    def get_pose_cam_in_world(self):
        return self.T_WC

    def get_pose_footprint_in_world(self):
        return self.T_WC

    def get_propropioceptive_state(self):
        return self.proprioceptive_state

    def is_valid(self):
        return isinstance(self.proprioceptive_state, torch.Tensor)


class LocalImageNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    name = "local_image_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        T_WB: torch.tensor = torch.eye(4),
        T_BC: torch.tensor = torch.eye(4),
        T_WC: torch.tensor = None,
        image: torch.tensor = None,
        projector: ImageProjector = None,
    ):
        assert isinstance(T_WB, torch.Tensor)
        assert isinstance(T_BC, torch.Tensor)
        super().__init__(timestamp, T_WB)

        self.T_BC = T_BC
        self.T_WC = self.T_WB @ self.T_BC if T_WC is None else T_WC
        self.image = image
        self.projector = projector

    def get_image(self):
        return self.image

    def get_pose_cam_in_world(self):
        return self.T_WC

    def get_image_projector(self):
        return self.projector

    def is_valid(self):
        return isinstance(self.image, torch.Tensor) and isinstance(self.projector, ImageProjector)


def run_base_state():
    """TODO."""

    import torch

    rs1 = BaseNode(1, T_WB=SE3(SO3.identity(), torch.Tensor([1, 0, 0])).as_matrix())
    rs2 = BaseNode(2, T_WB=SE3(SO3.identity(), torch.Tensor([2, 0, 0])).as_matrix())

    # Check that distance between robot states is correct
    assert abs(rs2.distance_to(rs1) - 1.0) < 1e-10

    # Check that objects are different
    assert rs1 != rs2

    # Check that timestamps are 1 second apart
    assert rs2.get_timestamp() - rs1.get_timestamp() == 1.0

    # Create node from another one
    rs3 = BaseNode.from_node(rs1)
    assert rs3 == rs1


if __name__ == "__main__":
    run_base_state()
