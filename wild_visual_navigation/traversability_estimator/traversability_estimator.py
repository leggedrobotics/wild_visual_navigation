from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.traversability_estimator import (
    LocalGraph,
    GlobalGraph,
    DebugNode,
    LocalImageNode,
    LocalProprioceptionNode,
)
import os
from os.path import join
import torch
import networkx as nx


class TraversabilityEstimator:
    def __init__(self, time_window):
        self.local_image_graph = LocalGraph(time_window)
        self.local_debug_graph = LocalGraph(time_window)
        self.local_proprio_graph = LocalGraph(time_window)

        self.global_graph = GlobalGraph()

    def add_local_image_node(self, node):
        if node.is_valid():
            # Add image node
            self.local_image_graph.add_node(node)

            # Add debug node
            debug_node = DebugNode.from_node(node)
            debug_node.set_traversability_mask(node.get_image())  # black image
            self.local_debug_graph.add_node(debug_node)

    def add_local_proprio_node(self, node):
        if node.is_valid():
            self.local_proprio_graph.add_node(node)

    def add_global_node(self, node):
        if node.is_valid():
            self.global_graph.add_node(node)

    def get_local_debug_nodes(self):
        return self.local_debug_graph.get_nodes()

    def get_local_image_nodes(self):
        return self.local_image_graph.get_nodes()

    def get_local_proprio_nodes(self):
        return self.local_proprio_graph.get_nodes()

    def train(self, iter=10):
        pass

    def update_labels(self, search_radius=None):
        """Iterates the nodes and projects their information from their neighbors

        Note: This is highly inefficient
        """
        print("update labels")

        # Iterate all nodes in the local graph
        for node, dnode in zip(self.local_image_graph.get_nodes(), self.local_debug_graph.get_nodes()):
            # Get neighbors
            proprio_nodes = self.local_proprio_graph.get_nodes_within_radius(node, radius=search_radius)
            # Get image projector
            image_projector = node.get_image_projector()
            # Get debug image
            traversability_mask = dnode.get_traversability_mask()
            # T_WC
            T_WC = node.get_pose_cam_in_world().unsqueeze(0)
            # print(str(node), T_WC[0, :3, 3])

            # Iterate neighbor proprioceptive nodes
            for ppnode in proprio_nodes:
                footprint = ppnode.get_footprint_points()
                color = torch.ones(1, 3)

                # Project and render mask
                _, traversability_mask_ = image_projector.project_and_render(
                    T_WC, footprint, color, image=traversability_mask
                )

            # Save traversability mask
            dnode.set_traversability_mask(traversability_mask)


def run_traversability_estimator():
    pass
