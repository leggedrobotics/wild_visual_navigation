from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.utils import make_box, make_rounded_box
from liegroups.torch import SE3, SO3
import torch


class BaseNode:
    """Base node data structure"""

    name = "base_node"

    def __init__(self, timestamp=0.0, T_WB=torch.eye(4)):
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
        return self.name == other.name and self.timestamp == other.timestamp

    def __lt__(self, other):
        self.timestamp < other.timestamp

    def is_valid(self):
        return (self.traversability is not None) and (self.image is not None) and (self.traversability_mask is not None)

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


class GlobalNode(BaseNode):
    """Global node stores the minimum information required for traversability estimation
    All the information is stored on the image plane"""

    name = "global_node"

    def __init__(self, timestamp=0.0, T_WB=torch.eye(4), features=None, supervision_signal=None):
        super().__init__(timestamp, T_WB)
        self.input_features = features
        self.supervision_signal = supervision_signal

    def get_features(self):
        return self.input_features

    def get_supervision_signal(self):
        return self.supervision_signal


class DebugNode(BaseNode):
    """Debug node stores images and full states for debugging"""

    name = "local_debug_node"

    def __init__(
        self, timestamp=0.0, T_WB=torch.eye(4), T_BC=torch.eye(4), T_BF=torch.eye(4), width=0.1, length=0.1, height=0.1
    ):
        super().__init__(timestamp, T_WB)

    def get_traversability_mask(self):
        return self.traversability_mask

    def set_traversability_mask(self, mask):
        self.traversability_mask = mask


class LocalProprioceptionNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    name = "local_proprioception_node"

    def __init__(
        self,
        timestamp=0.0,
        T_WB=torch.eye(4),
        T_BF=torch.eye(4),
        width=0.1,
        length=0.1,
        height=0.1,
        proprioception=None,
    ):
        assert isinstance(T_WB, torch.Tensor)
        assert isinstance(T_BF, torch.Tensor)
        super().__init__(timestamp, T_WB)

        self.T_BF = T_BF
        self.T_WF = self.T_WB @ self.T_BF  # footprint in world
        self.width = width
        self.length = length
        self.height = height
        self.proprioceptive_state = proprioception

    def get_bounding_box_points(self):
        return make_rounded_box(self.length, self.width, self.height, pose=self.T_WB)

    def get_footprint_points(self):
        return make_rounded_box(self.length, self.width, 0.0, pose=self.T_WF)

    def get_image(self):
        return self.image

    def get_pose_cam_in_world(self):
        return self.T_WC

    def get_pose_footprint_in_world(self):
        return self.T_WC

    def get_propropioceptive_state(self):
        return self.proprioceptive_state


class LocalImageNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    name = "local_image_node"

    def __init__(self, timestamp=0.0, T_WB=torch.eye(4), T_BC=torch.eye(4), image=None, projector=None):
        assert isinstance(T_WB, torch.Tensor)
        assert isinstance(T_BF, torch.Tensor)

        self.T_BC = T_BC
        self.T_WC = self.T_WB @ self.T_BC  # footprint in world
        self.image = image
        self.projector = projector

    def get_image(self):
        return self.image

    def get_pose_cam_in_world(self):
        return self.T_WC

    def get_image_projector(self):
        return self.projector


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


if __name__ == "__main__":
    run_base_state()