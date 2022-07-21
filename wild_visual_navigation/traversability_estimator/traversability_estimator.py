from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.traversability_estimator import (
    BaseNode,
    BaseGraph,
    DistanceWindowGraph,
    GlobalNode,
    ImageNode,
    ProprioceptionNode,
)

from threading import Thread, Lock
import os
import torch
import networkx as nx
import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()

# debug
from wild_visual_navigation.utils import get_img_from_fig
import matplotlib.pyplot as plt
from kornia.utils import tensor_to_image
from stego.src import remove_axes


class TraversabilityEstimator:
    def __init__(
        self,
        device: str = "cuda",
        max_distance: float = 3,
        image_distance_thr: float = None,
        proprio_distance_thr: float = None,
        feature_extractor: str = "dino_slic",
    ):
        self.device = device
        # Local graphs
        self.image_graph = DistanceWindowGraph(max_distance=max_distance, edge_distance=image_distance_thr)
        self.proprio_graph = DistanceWindowGraph(max_distance=max_distance, edge_distance=proprio_distance_thr)
        # Experience graph
        self.experience_graph = BaseGraph()

        # TODO: fix feature extractor type
        self.feature_extractor = FeatureExtractor(device, extractor_type=feature_extractor)
        # For debugging
        os.makedirs(os.path.join(WVN_ROOT_DIR, "results", "test_traversability_estimator"), exist_ok=True)

        # Mutex
        self.lock = Lock()

    def add_image_node(self, node: BaseNode):
        """Adds a node to the local graph to store images

        Args:
            node (BaseNode): new node in the image graph
        """

        if node.is_valid():
            # Add image node
            if self.image_graph.add_node(node):
                # Add global node
                global_node = GlobalNode.from_node(node)
                global_node.set_image(node.get_image())
                self.experience_graph.add_node(global_node)
                print(f"Adding new node {global_node}")

                # Project past footprints on current image
                image_projector = node.get_image_projector()
                T_WC = node.get_pose_cam_in_world().unsqueeze(0)
                supervision_mask = node.get_image() * 0

                for pnode in self.proprio_graph.get_nodes():
                    footprint = pnode.get_footprint_points().unsqueeze(0)
                    color = torch.FloatTensor([1.0, 1.0, 1.0])
                    # Project and render mask
                    mask, _ = image_projector.project_and_render(T_WC, footprint, color)
                    mask = mask.squeeze(0)
                    # Update supervision mask
                    supervision_mask = torch.maximum(supervision_mask, mask.to(supervision_mask.device))

                global_node.set_supervision_mask(supervision_mask)

    def add_proprio_node(self, node: BaseNode):
        """Adds a node to the local graph to store proprioception

        Args:
            node (BaseNode): new node in the proprioceptive graph
        """
        if not node.is_valid():
            return False

        if not self.proprio_graph.add_node(node):
            return False

        else:
            # Get proprioceptive information
            footprint = node.get_footprint_points().unsqueeze(0)
            color = torch.FloatTensor([1.0, 1.0, 1.0])

            # Project footprint onto all the image nodes
            for inode in self.image_graph.get_nodes():
                # Get global node
                global_node = self.experience_graph.get_node_with_timestamp(inode.get_timestamp())
                if global_node is None:
                    continue

                # Get stuff from image node
                image_projector = inode.get_image_projector()
                T_WC = inode.get_pose_cam_in_world().unsqueeze(0)
                # Get stuff from global node
                supervision_mask = global_node.get_supervision_mask()

                # Project and render mask
                mask, _ = image_projector.project_and_render(T_WC, footprint, color)

                if mask is None or supervision_mask is None:
                    continue

                # Update traversability mask
                mask = mask.squeeze(0)
                supervision_mask = torch.maximum(supervision_mask, mask.to(supervision_mask.device))

                # Get global node and update supervision signal
                global_node.set_supervision_mask(supervision_mask)

            return True

    def get_image_nodes(self):
        return self.image_graph.get_nodes()

    def get_proprio_nodes(self):
        return self.proprio_graph.get_nodes()

    def get_last_valid_global_node(self):
        # last_image_node = self.image_graph.get_nodes()[0]
        # last_global_node = self.experience_graph.get_node_with_timestamp(last_image_node.get_timestamp())
        # return last_global_node if last_global_node.is_valid() else None
        last_valid_node = None
        for node in self.experience_graph.get_nodes():
            if node.is_valid():
                last_valid_node = node
        return last_valid_node

    def save_graph(self, mission_path: str, export_debug: bool = False):
        # Make folder if it doesn't exist
        os.makedirs(mission_path, exist_ok=True)

        # Get all the current nodes
        global_nodes = self.experience_graph.get_nodes()
        for node, index in zip(global_nodes, range(len(global_nodes))):
            node.save(mission_path, index)

    def train(self, iter=10):
        pass

    def update_features(self):
        for node in self.experience_graph.get_nodes():
            # print(f"updating node {node}")

            # Update features
            if node.get_features() is None:
                print(f"updating features in {node}")
                # Run feature extractor
                edges, feat, seg, center = self.feature_extractor.extract(
                    img=node.get_image().clone().unsqueeze(0), return_centers=True
                )

                # Set features in global graph
                node.set_features(
                    feature_type=self.feature_extractor.get_type(),
                    features=feat,
                    edges=edges,
                    segments=seg,
                    positions=center,
                )

            # Update supervision signal
            print(f"updating supervision in {node}")
            node.update_supervision_signal()

    # def update_labels_and_features(self, search_radius: float = None):
    #     """Iterates the nodes and projects their information from their neighbors
    #     Note: This is highly inefficient

    #     Args:
    #         search_radius (float): to find neighbors in the graph
    #     """
    #     # Iterate all nodes in the local graph
    #     for node, dnode, gnode in zip(
    #         self.image_graph.get_nodes(), self.debug_graph.get_nodes(), self.experience_graph.get_nodes()
    #     ):
    #         # Get neighbors
    #         proprio_nodes = self.proprio_graph.get_nodes_within_radius_range(node, min_radius = 0.01, max_radius=search_radius, time_eps=5)

    #         # Get image projector
    #         image_projector = node.get_image_projector()
    #         # T_WC
    #         T_WC = node.get_pose_cam_in_world().unsqueeze(0)
    #         # Get debug image
    #         image = node.get_image()
    #         supervision_mask = dnode.get_supervision_mask()

    #         # Compute features if we haven't done it
    #         if dnode.get_image() is None:
    #             # Run feature extractor
    #             edges, feat, seg, center = self.feature_extractor.extract(
    #                 img=image.clone().unsqueeze(0), return_centers=True
    #             )

    #             # Set features in global graph
    #             gnode.set_features(
    #                 feature_type=self.feature_extractor.get_type(),
    #                 features=feat,
    #                 edges=edges,
    #                 segments=seg,
    #                 positions=center,
    #             )

    #             # Set image in debug graph
    #             dnode.set_image(image)

    #         # Iterate neighbor proprioceptive nodes
    #         for ppnode in proprio_nodes:
    #             footprint = ppnode.get_footprint_points().unsqueeze(0)
    #             color = torch.FloatTensor([1.0, 1.0, 1.0])

    #             # Project and render mask
    #             mask, _ = image_projector.project_and_render(T_WC, footprint, color)
    #             mask = mask.squeeze(0)

    #             # Update traversability mask
    #             supervision_mask = torch.maximum(supervision_mask, mask.to(supervision_mask.device))

    #         # Save traversability in global node to store supervision signal
    #         gnode.set_supervision_signal(supervision_mask, is_image=True)

    #         # Save traversability mask and labeled image in debug node
    #         with self.debug_graph.lock:
    #             dnode.set_supervision_mask(supervision_mask)
    #             dnode.set_training_node(gnode)


def run_traversability_estimator():
    pass
