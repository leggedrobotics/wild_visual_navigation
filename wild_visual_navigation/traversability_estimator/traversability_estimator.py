from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.traversability_estimator import LocalGraph, GlobalGraph
import os
from os.path import join
import torch
import networkx as nx


class TraversabilityEstimator:
    def __init__(self, image_projector, time_window):
        self.image_projector = image_projector
        self.local_graph = LocalGraph(time_window)
        self.global_graph = GlobalGraph()

    def add_local_node(self, node):
        if node.is_valid():
            self.local_graph.add_node(node)

    def add_global_node(self, node):
        if node.is_valid():
            self.global_graph.add_node(node)

    def train(self, iter=10):
        pass

    def update_labels(self, search_radius=None):
        """Iterates the nodes and projects their information from their neighbors

        Note: This is highly inefficient
        """

        # Iterate all nodes in the local graph
        for node in self.local_graph.get_nodes():

            # Get neighbors
            if search_radius is None:
                neighbors = self.graph.get_nodes()
            else:
                neighbors = self.graph.get_nodes_within_radius(node, search_radius)

            # get traversability mask
            traversability_mask = node.get_traversability_mask()

            # get image projector
            image_projector = node.get_projector()

            # Iterate neighbors
            neigh_masks = []
            for neigh in neighbors:
                footprint = neigh.get_footprint_points()
                color = torch.ones(1, 3) * neig.get_traversability()

                # Project and render mask
                traversability_mask_, _ = image_projector.project_and_render(
                    node.get_pose_cam_in_world(), footprint, "world", color
                )

                # Save neighbor masks
                neigh_masks.append(traversability_mask)

            # Combine traversability masks

            # Save traversability mask
            node.set_traversability_mask(traversability_mask)


def run_traversability_estimator():
    pass
