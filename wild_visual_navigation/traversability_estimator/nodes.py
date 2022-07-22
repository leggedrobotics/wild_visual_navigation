from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.utils import make_box, make_rounded_box, make_plane
from liegroups.torch import SE3, SO3
from PIL import Image, ImageDraw
from skimage import segmentation
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
    ):
        super().__init__(timestamp=timestamp, T_WB=T_WB)
        self.features = None
        self.feature_type = None
        self.feature_edges = None
        self.feature_segments = None
        self.feature_positions = None
        self.image = None
        self.supervision_mask = None
        self.supervision_signal = None
        self.supervision_signal_valid = None

    def as_pyg_data(self):
        return Data(
            x=self.features,
            edge_index=self.feature_edges,
            y=self.supervision_signal,
            y_valid=self.supervision_signal_valid,
        )

    def is_valid(self):
        return isinstance(self.features, torch.Tensor) and isinstance(self.supervision_signal, torch.Tensor)

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

    def get_image(self):
        return self.image

    def get_supervision_signal(self):
        return self.supervision_signal

    def get_supervision_mask(self):
        return self.supervision_mask

    def get_training_image(self):
        if self.image is None or self.supervision_mask is None:
            return None
        img_np = kornia.utils.tensor_to_image(self.image)
        trav_np = kornia.utils.tensor_to_image(self.supervision_mask)

        # Draw segments
        # trav_np = segmentation.mark_boundaries(trav_np, self.feature_segments.cpu().numpy()[0,0])
        img_np = segmentation.mark_boundaries(img_np, self.feature_segments.cpu().numpy())

        img_pil = Image.fromarray(np.uint8(img_np * 255))
        img_draw = ImageDraw.Draw(img_pil)
        trav_pil = Image.fromarray(np.uint8(trav_np * 255))

        # Draw graph
        for i in range(self.feature_edges.shape[1]):
            a, b = self.feature_edges[0, i], self.feature_edges[1, i]
            line_params = self.feature_positions[a].tolist() + self.feature_positions[b].tolist()
            img_draw.line(line_params, fill=(255, 255, 255, 100), width=2)

        for i in range(self.feature_positions.shape[1]):
            params = self.feature_positions[i].tolist()
            color = trav_pil.getpixel((params[0], params[1]))
            r = 10
            params = [p - r for p in params] + [p + r for p in params]
            img_draw.ellipse(params, fill=color)

        np_draw = np.array(img_pil)
        return kornia.utils.image_to_tensor(np_draw.copy()).to(self.image.device)

    def save(self, output_path: str, index: int, graph_only: bool = False):
        graph_data = self.as_pyg_data()
        path = os.path.join(output_path, "graph", f"graph_{index:06d}.pt")
        torch.save(graph_data, path)
        if not graph_only:
            p = path.replace("graph", "img")
            torch.save(self.image.cpu(), p)

            p = path.replace("graph", "center")
            torch.save(self.feature_positions.cpu(), p)

            p = path.replace("graph", "seg")
            torch.save(self.feature_segments.cpu(), p)

    def set_image(self, image: torch.tensor):
        self.image = image

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

    def set_supervision_mask(self, mask: torch.tensor):
        self.supervision_mask = mask

    def update_supervision_signal(self):
        if self.supervision_mask is None:
            return

        if len(self.supervision_mask.shape) == 3:
            signal = self.supervision_mask.mean(axis=0)

        # If we don't have features, return
        if self.features is None:
            return

        # If we have features, update supervision signal
        labels_per_segment = []
        for s in range(self.feature_segments.max() + 1):
            # Get a mask indices for the segment
            m = self.feature_segments == s
            # Add the higehst number per segment
            labels_per_segment.append(signal[m].max())

        # Prepare supervision signal
        torch_labels = torch.stack(labels_per_segment)
        if torch_labels.sum() > 0:
            self.supervision_signal = torch_labels
            # Binary mask
            self.supervision_signal_valid = torch_labels > 0


class ProprioceptionNode(BaseNode):
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

    def get_pose_cam_in_world(self):
        return self.T_WC

    def get_pose_footprint_in_world(self):
        return self.T_WC

    def get_propropioceptive_state(self):
        return self.proprioceptive_state

    def is_valid(self):
        return isinstance(self.proprioceptive_state, torch.Tensor)


class ImageNode(BaseNode):
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
