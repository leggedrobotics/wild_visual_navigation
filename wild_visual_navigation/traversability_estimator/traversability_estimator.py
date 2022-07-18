from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.traversability_estimator import (
    BaseNode,
    LocalGraph,
    GlobalGraph,
    DebugNode,
    GlobalNode,
    LocalImageNode,
    LocalProprioceptionNode,
)

from threading import Thread

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
        time_window: float = 10,
        image_distance_thr: float = None,
        proprio_distance_thr: float = None,
        feature_extractor: str = "dino_slic",
    ):
        self.device = device
        # Local graphs
        self.local_image_graph = LocalGraph(time_window, edge_distance=image_distance_thr)
        self.local_debug_graph = LocalGraph(time_window, edge_distance=image_distance_thr)
        self.local_proprio_graph = LocalGraph(time_window, edge_distance=proprio_distance_thr)
        # Global graph
        self.global_graph = GlobalGraph()
        # TODO: fix feature extractor type
        self.feature_extractor = FeatureExtractor(device, extractor_type=feature_extractor)
        # For debugging
        os.makedirs(os.path.join(WVN_ROOT_DIR, "results", "test_traversability_estimator"), exist_ok=True)

    def add_local_image_node(self, node: BaseNode):
        """Adds a node to the local graph to store images

        Args:
            node (BaseNode): new node in the image graph
        """

        if node.is_valid():
            # Add image node
            self.local_image_graph.add_node(node)

            # Add debug node
            debug_node = DebugNode.from_node(node)
            debug_node.set_traversability_mask(node.get_image() * 0)
            self.local_debug_graph.add_node(debug_node)

            # Add global node
            global_node = GlobalNode.from_node(node)
            self.global_graph.add_node(global_node)

    def add_local_proprio_node(self, node: BaseNode):
        """Adds a node to the local graph to store proprioception

        Args:
            node (BaseNode): new node in the proprioceptive graph
        """
        if node.is_valid():
            return self.local_proprio_graph.add_node(node)

    def add_global_node(self, node: BaseNode):
        """Adds a node to the global graph to store training data

        Args:
            node (BaseNode): new node in the image graph
        """
        if node.is_valid():
            return self.global_graph.add_node(node)

    def get_local_debug_nodes(self):
        return self.local_debug_graph.get_nodes()

    def get_local_image_nodes(self):
        return self.local_image_graph.get_nodes()

    def get_local_proprio_nodes(self):
        return self.local_proprio_graph.get_nodes()

    def save_graph(self, mission_path: str, export_debug: bool = False):
        # Make folder if it doesn't exist
        os.makedirs(mission_path, exist_ok=True)

        # Get all the current nodes
        global_nodes = self.global_graph.get_nodes()
        for node, index in zip(global_nodes, range(len(global_nodes))):
            node.save(mission_path, index)        
        

    def train(self, iter=10):
        pass

    def update_labels_and_features(self, search_radius: float = None):
        """Iterates the nodes and projects their information from their neighbors
        Note: This is highly inefficient

        Args:
            search_radius (float): to find neighbors in the graph
        """
        # Iterate all nodes in the local graph
        for node, dnode, gnode in zip(
            self.local_image_graph.get_nodes(), self.local_debug_graph.get_nodes(), self.global_graph.get_nodes()
        ):
            # Get neighbors
            proprio_nodes = self.local_proprio_graph.get_nodes_within_radius(node, radius=search_radius, time_eps=5)

            # Get image projector
            image_projector = node.get_image_projector()
            # T_WC
            T_WC = node.get_pose_cam_in_world().unsqueeze(0)
            # Get debug image
            image = node.get_image()
            traversability_mask = dnode.get_traversability_mask()

            # Compute features if we haven't done it
            if dnode.get_image() is None:
                # Run feature extractor
                edges, feat, seg, center = self.feature_extractor.extract(
                    img=image.clone().unsqueeze(0), return_centers=True
                )

                # Set features in global graph
                gnode.set_features(
                    feature_type=self.feature_extractor.get_type(),
                    features=feat,
                    edges=edges,
                    segments=seg,
                    positions=center,
                )

                # Set features in debug graph
                dnode.set_image(image)

            # Iterate neighbor proprioceptive nodes
            for ppnode in proprio_nodes:
                footprint = ppnode.get_footprint_points().unsqueeze(0)
                color = torch.FloatTensor([1.0, 1.0, 1.0])

                # Project and render mask
                mask, _ = image_projector.project_and_render(T_WC, footprint, color)
                mask = mask.squeeze(0)

                # Update traversability mask
                traversability_mask = torch.maximum(traversability_mask, mask.to(traversability_mask.device))


            # Save traversability in global node to store supervision signal
            gnode.set_supervision_signal(traversability_mask, is_image=True)

            # Save traversability mask and labeled image in debug node
            with self.local_debug_graph.lock:
                dnode.set_traversability_mask(traversability_mask)
                dnode.set_training_node(gnode)


def run_traversability_estimator():
    pass
