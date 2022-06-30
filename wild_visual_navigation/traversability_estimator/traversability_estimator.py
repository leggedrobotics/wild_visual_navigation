from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.traversability_estimator import (
    LocalGraph,
    GlobalGraph,
    DebugNode,
    GlobalNode,
    LocalImageNode,
    LocalProprioceptionNode,
)
import os
from os.path import join
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
    def __init__(self, device="cuda", time_window=10, image_distance_thr=None, proprio_distance_thr=None):
        self.device = device
        # Local graphs
        self.local_image_graph = LocalGraph(time_window, edge_distance=image_distance_thr)
        self.local_debug_graph = LocalGraph(time_window, edge_distance=image_distance_thr)
        self.local_proprio_graph = LocalGraph(time_window, edge_distance=proprio_distance_thr)
        # Global graph
        self.global_graph = GlobalGraph()
        # Feature extractor
        self.feature_extractor = FeatureExtractor(device)
        # For debugging
        os.makedirs(join(WVN_ROOT_DIR, "results", "test_traversability_estimator"), exist_ok=True)

    def add_local_image_node(self, node):
        if node.is_valid():
            # Add image node
            self.local_image_graph.add_node(node)

            # Add debug node
            debug_node = DebugNode.from_node(node)
            debug_node.set_traversability_mask(node.get_image() * 0)
            debug_node.set_labeled_image(node.get_image())
            self.local_debug_graph.add_node(debug_node)

            # Add global node
            global_node = GlobalNode.from_node(node)
            self.global_graph.add_node(global_node)

    def add_local_proprio_node(self, node):
        if node.is_valid():
            return self.local_proprio_graph.add_node(node)

    def add_global_node(self, node):
        if node.is_valid():
            return self.global_graph.add_node(node)

    def get_local_debug_nodes(self):
        return self.local_debug_graph.get_nodes()

    def get_local_image_nodes(self):
        return self.local_image_graph.get_nodes()

    def get_local_proprio_nodes(self):
        return self.local_proprio_graph.get_nodes()

    def train(self, iter=10):
        pass

    def update_labels_and_features(self, search_radius=None):
        """Iterates the nodes and projects their information from their neighbors

        Note: This is highly inefficient
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
            labeled_image = dnode.get_labeled_image()

            # Compute features if we haven't done it
            if dnode.get_features_image() is None:
                # DINO default feature extractor
                # TODO: this should return a torch image
                adj, feat, seg, center, img = self.feature_extractor.dino_slic(
                    image.clone().unsqueeze(0), return_centers=True, return_image=True
                )
                gnode.set_features(feat)
                dnode.set_features_image(to_tensor(img))

            # Iterate neighbor proprioceptive nodes
            for ppnode in proprio_nodes:
                footprint = ppnode.get_footprint_points().unsqueeze(0)
                color = torch.FloatTensor([0.0, 1.0, 0.0])

                # Project and render mask
                mask, labeled_image = image_projector.project_and_render(T_WC, footprint, color, image=labeled_image)
                mask = mask.squeeze(0)
                labeled_image = labeled_image.squeeze(0)

                # Update traversability mask
                traversability_mask = torch.maximum(traversability_mask, mask.to(traversability_mask.device))

            # Save traversability mask and labeled image in debug node
            with self.local_debug_graph.lock:
                dnode.set_traversability_mask(traversability_mask)
                dnode.set_labeled_image(labeled_image)

            # # Plot result as in colab
            # fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5))
            # ax[0].imshow(tensor_to_image(labeled_image[[2, 1, 0], :, :]))
            # ax[0].set_title("Image")
            # ax[1].imshow(tensor_to_image(traversability_mask[[2, 1, 0], :, :]))
            # ax[1].set_title("Mask")
            # remove_axes(ax)
            # plt.tight_layout()

            # # Store results to test directory
            # img = get_img_from_fig(fig)
            # img.save(join(WVN_ROOT_DIR, "results", "test_traversability_estimator", f"{str(node)}.png"))


def run_traversability_estimator():
    pass
