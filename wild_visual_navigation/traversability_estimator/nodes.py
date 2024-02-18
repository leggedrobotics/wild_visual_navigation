#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.utils import (
    make_box,
    make_plane,
    make_polygon_from_points,
    make_dense_plane,
)
from liegroups.torch import SE3, SO3
from wild_visual_navigation.utils import Data

import os
import torch
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
            SE3.from_matrix(
                self.pose_base_in_world.inverse() @ other.pose_base_in_world,
                normalize=True,
            )
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
        camera_name="cam",
        use_for_training=True,
    ):
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)
        # Initialize members
        self._pose_cam_in_base = pose_cam_in_base
        self._pose_cam_in_world = (
            self._pose_base_in_world @ self._pose_cam_in_base if pose_cam_in_world is None else pose_cam_in_world
        )
        self._image = image
        self._image_projector = image_projector
        self._camera_name = camera_name
        self._use_for_training = use_for_training

        # Uninitialized members
        self._features = None
        self._feature_edges = None
        self._feature_segments = None
        self._feature_positions = None
        self._prediction = None
        self._supervision_mask = None
        self._supervision_signal = None
        self._supervision_signal_valid = None
        self._confidence = None

    def clear_debug_data(self):
        """Removes all data not required for training"""
        try:
            del self._image
            del self._supervision_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            pass  # Image already removed

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._image_projector.change_device(device)

        self._pose_cam_in_base = self._pose_cam_in_base.to(device)
        self._pose_cam_in_world = self._pose_cam_in_world.to(device)

        if self._image is not None:
            self._image = self._image.to(device)
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
        if self._confidence is not None:
            self._confidence = self._confidence.to(device)

    def as_pyg_data(
        self,
        previous_node: Optional[BaseNode] = None,
        anomaly_detection: bool = False,
        aux: bool = False,
    ):
        if aux:
            return Data(x=self.features, edge_index=self._feature_edges)
        if previous_node is None:
            if anomaly_detection:
                return Data(
                    x=self.features[self._supervision_signal_valid],
                    edge_index=self._feature_edges,
                    y=self._supervision_signal[self._supervision_signal_valid],
                    y_valid=self._supervision_signal_valid[self._supervision_signal_valid],
                )
            else:
                return Data(
                    x=self.features,
                    edge_index=self._feature_edges,
                    y=self._supervision_signal,
                    y_valid=self._supervision_signal_valid,
                )

        else:
            if anomaly_detection:
                return Data(
                    x=self.features[self._supervision_signal_valid],
                    edge_index=self._feature_edges,
                    y=self._supervision_signal[self._supervision_signal_valid],
                    y_valid=self._supervision_signal_valid[self._supervision_signal_valid],
                    x_previous=previous_node.features,
                    edge_index_previous=previous_node._feature_edges,
                )
            else:
                return Data(
                    x=self.features,
                    edge_index=self._feature_edges,
                    y=self._supervision_signal,
                    y_valid=self._supervision_signal_valid,
                    x_previous=previous_node.features,
                    edge_index_previous=previous_node._feature_edges,
                )

    def is_valid(self):
        valid_members = (
            isinstance(self._features, torch.Tensor)
            and isinstance(self._supervision_signal, torch.Tensor)
            and isinstance(self._supervision_signal_valid, torch.Tensor)
        )
        valid_signals = self._supervision_signal_valid.any() if valid_members else False

        return valid_members and valid_signals

    @property
    def camera_name(self):
        return self._camera_name

    @property
    def confidence(self):
        return self._confidence

    @property
    def features(self):
        return self._features

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
        return self._prediction

    @property
    def supervision_signal(self):
        return self._supervision_signal

    @property
    def supervision_signal_valid(self):
        return self._supervision_signal_valid

    @property
    def supervision_mask(self):
        return self._supervision_mask

    @property
    def use_for_training(self):
        return self._use_for_training

    @camera_name.setter
    def camera_name(self, camera_name):
        self._camera_name = camera_name

    @confidence.setter
    def confidence(self, confidence):
        self._confidence = confidence

    @features.setter
    def features(self, features):
        self._features = features

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

    @supervision_signal_valid.setter
    def supervision_signal_valid(self, _supervision_signal_valid):
        self._supervision_signal_valid = _supervision_signal_valid

    @supervision_mask.setter
    def supervision_mask(self, supervision_mask):
        self._supervision_mask = supervision_mask

    @use_for_training.setter
    def use_for_training(self, use_for_training):
        self._use_for_training = use_for_training

    def save(
        self,
        output_path: str,
        index: int,
        graph_only: bool = False,
        previous_node: Optional[BaseNode] = None,
    ):
        if self._feature_positions is not None:
            graph_data = self.as_pyg_data(previous_node)
            path = os.path.join(output_path, "graph", f"graph_{index:06d}.pt")
            torch.save(graph_data, path)
            if not graph_only:
                p = path.replace("graph", "img")
                torch.save(self._image.cpu(), p)

                p = path.replace("graph", "center")
                torch.save(self._feature_positions.cpu(), p)

                p = path.replace("graph", "seg")
                torch.save(self._feature_segments.cpu(), p)

    def project_footprint(
        self,
        footprint: torch.tensor,
        color: torch.tensor = torch.FloatTensor([1.0, 1.0, 1.0]),
    ):
        (
            mask,
            image_overlay,
            projected_points,
            valid_points,
        ) = self._image_projector.project_and_render(self._pose_cam_in_world[None], footprint, color)

        return mask, image_overlay, projected_points, valid_points

    def update_supervision_signal(self):
        if self._supervision_mask is None:
            return

        if len(self._supervision_mask.shape) == 3:
            signal = self._supervision_mask.nanmean(axis=0)

        # If we don't have features, return
        if self._features is None:
            return

        # If we have features, update supervision signal
        N, M = signal.shape
        num_segments = self._feature_segments.max() + 1
        torch.arange(0, num_segments)[None, None]

        # Create array to mask by index (used to select the segments)
        multichannel_index_mask = torch.arange(0, num_segments, device=self._feature_segments.device)[
            None, None
        ].expand(N, M, num_segments)
        # Make a copy of the segments with the dimensionality of the segments, so we can split them on each channel
        multichannel_segments = self._feature_segments[:, :, None].expand(N, M, num_segments)

        # Create a multichannel mask that allows to associate a segment to each channel
        multichannel_segments_mask = multichannel_index_mask == multichannel_segments

        # Apply the mask to an expanded supervision signal and get the mean value per segment
        # First we get the number of elements per segment (stored on each channel)
        num_elements_per_segment = (
            multichannel_segments_mask * ~torch.isnan(signal[:, :, None].expand(N, M, num_segments))
        ).sum(dim=[0, 1])
        # We get the sum of all the values of the supervision signal that fall in the segment
        signal_sum = (signal.nan_to_num(0)[:, :, None].expand(N, M, num_segments) * multichannel_segments_mask).sum(
            dim=[0, 1]
        )
        # Compute the average of the supervision signal dividing by the number of elements
        signal_mean = signal_sum / num_elements_per_segment

        # Finally replace the nan values to 0.0
        self._supervision_signal = signal_mean.nan_to_num(0)
        self._supervision_signal_valid = self._supervision_signal > 0


class SupervisionNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency
    """

    _name = "supervision_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_footprint_in_base: torch.tensor = torch.eye(4),
        pose_footprint_in_world: torch.tensor = None,
        twist_in_base: torch.tensor = None,
        desired_twist_in_base: torch.tensor = None,
        length: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
        supervision: torch.tensor = None,
        traversability: torch.tensor = torch.FloatTensor([0.0]),
        traversability_var: torch.tensor = torch.FloatTensor([1.0]),
        is_untraversable: bool = False,
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
        self._twist_in_base = twist_in_base
        self._desired_twist_in_base = desired_twist_in_base
        self._length = length
        self._width = width
        self._height = height
        self._supervision_state = supervision
        self._traversability = traversability
        self._traversability_var = traversability_var
        self._is_untraversable = is_untraversable

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._pose_footprint_in_base = self._pose_footprint_in_base.to(device)
        self._pose_footprint_in_world = self._pose_footprint_in_world.to(device)
        self._twist_in_base = self._twist_in_base.to(device)
        self._desired_twist_in_base = self._desired_twist_in_base.to(device)
        self._supervision_state = self._supervision_state.to(device)

    def get_bounding_box_points(self):
        return make_box(
            self._length,
            self._width,
            self._height,
            pose=self._pose_base_in_world,
            grid_size=5,
        ).to(self._pose_base_in_world.device)

    def get_footprint_points(self):
        return make_plane(
            x=self._length,
            y=self._width,
            pose=self._pose_footprint_in_world,
            grid_size=25,
        ).to(self._pose_footprint_in_world.device)

    def get_side_points(self):
        return make_plane(x=0.0, y=self._width, pose=self._pose_footprint_in_world, grid_size=2).to(
            self._pose_footprint_in_world.device
        )

    def get_untraversable_plane(self, grid_size=5):
        device = self._pose_footprint_in_world.device
        motion_direction = self._twist_in_base / self._twist_in_base.norm()

        # dim_twist = motion_direction.shape[-1]
        # if dim_twist != 2:
        #     print(f"Warning: input twist has dimension [{dim_twist}], will assume that twist[0]=vx, twist[1]=vy")

        # Compute angle of motion
        z_angle = torch.atan2(motion_direction[1], motion_direction[0]).item()

        # Prepare transformation of plane in base frame
        rho = torch.FloatTensor(
            [
                0.5 * self._length * motion_direction[0],
                0.5 * self._length * motion_direction[1],
                -self._height / 2,
            ]
        )  # Translation vector (x, y, z)
        phi = torch.FloatTensor([0.0, 0.0, z_angle])  # roll-pitch-yaw
        R_BP = SO3.from_rpy(phi)
        pose_plane_in_base = SE3(R_BP, rho).as_matrix().to(device)  # Pose matrix of plane in base frame
        pose_plane_in_world = self._pose_base_in_world @ pose_plane_in_base  # Pose of plane in world frame

        # Make plane
        return make_dense_plane(
            y=0.5 * self._width,
            z=self._height,
            pose=pose_plane_in_world,
            grid_size=grid_size,
        ).to(device)

    def make_footprint_with_node(self, other: BaseNode, grid_size: int = 10):
        if self.is_untraversable:
            footprint = self.get_untraversable_plane(grid_size=grid_size)
        else:
            # Get side points
            other_side_points = other.get_side_points()
            this_side_points = self.get_side_points()
            # swap points to make them counterclockwise
            this_side_points[[0, 1]] = this_side_points[[1, 0]]
            # The idea is to make a polygon like:
            # tsp[1] ---- tsp[0]
            #  |            |
            # osp[0] ---- osp[1]
            # with 'tsp': this_side_points and 'osp': other_side_points

            # Concat points to define the polygon
            points = torch.concat((this_side_points, other_side_points), dim=0)
            # Make footprint
            footprint = make_polygon_from_points(points, grid_size=grid_size)
        return footprint

    def update_traversability(self, traversability: torch.tensor, traversability_var: torch.tensor):
        # Pessimistic rule: choose the less traversable one
        if (traversability < self._traversability).any():
            self._traversability = traversability
            self._traversability_var = traversability_var

    @property
    def traversability(self):
        return self._traversability

    @property
    def traversability_var(self):
        return self._traversability_var

    @property
    def twist_in_base(self):
        return self._twist_in_base

    @property
    def desired_twist_in_base(self):
        return self._desired_twist_in_base

    @property
    def is_untraversable(self):
        return self._is_untraversable

    @property
    def pose_footprint_in_world(self):
        return self._pose_footprint_in_world

    @property
    def supervision_state(self):
        return self._supervision_state

    @traversability.setter
    def traversability(self, traversability):
        self._traversability = traversability

    @traversability_var.setter
    def traversability_var(self, variance):
        self._traversability_var = variance

    def is_valid(self):
        return isinstance(self._supervision_state, torch.Tensor)


class TwistNode(BaseNode):
    """Stores twist information"""

    _name = "twist_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        desired_twist: torch.tensor = torch.zeros(6),
        current_twist: torch.tensor = torch.zeros(6),
    ):
        assert isinstance(pose_base_in_world, torch.Tensor)
        assert isinstance(desired_twist, torch.Tensor)
        assert isinstance(current_twist, torch.Tensor)
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)

        self._desired_twist = desired_twist
        self._current_twist = current_twist

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._desired_twist = self._desired_twist.to(device)
        self._current_twist = self._current_twist.to(device)

    @property
    def desired_twist(self):
        return self._desired_twist

    @property
    def current_twist(self):
        return self._current_twist

    @desired_twist.setter
    def desired_twist(self, desired_twist):
        self._desired_twist = desired_twist

    @current_twist.setter
    def current_twist(self, current_twist):
        self._current_twist = current_twist


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
