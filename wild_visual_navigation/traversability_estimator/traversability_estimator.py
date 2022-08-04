from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.learning.model import get_model
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.traversability_estimator import (
    BaseGraph,
    DistanceWindowGraph,
    MissionNode,
    ProprioceptionNode,
)
from wild_visual_navigation.utils import make_polygon_from_points
from wild_visual_navigation.visu import LearningVisualizer
from pytorch_lightning import seed_everything
from torch_geometric.data import Data, Batch
from threading import Lock
import dataclasses
import os
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()


class TraversabilityEstimator:
    def __init__(
        self,
        device: str = "cuda",
        max_distance: float = 3,
        image_distance_thr: float = None,
        proprio_distance_thr: float = None,
        feature_extractor_type: str = "dino_slic",
        min_samples_for_training: int = 10,
    ):
        self.device = device
        self.min_samples_for_training = min_samples_for_training

        # Local graphs
        self._proprio_graph = DistanceWindowGraph(max_distance=max_distance, edge_distance=proprio_distance_thr)
        # Experience graph
        self._mission_graph = BaseGraph()

        # Feature extractor
        self._feature_extractor_type = feature_extractor_type
        self._feature_extractor = FeatureExtractor(device, extractor_type=self._feature_extractor_type)

        # Mutex
        self._lock = Lock()
        self._pause_training = False

        # Visualization
        self._visualizer = LearningVisualizer()

        # Lightning module
        seed_everything(42)
        self._exp_cfg = dataclasses.asdict(ExperimentParams())

        self._model = get_model(self._exp_cfg["model"]).to(device)
        self._epoch = 0
        self._last_trained_model = self._model.to(device)
        self._model.train()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._exp_cfg["optimizer"]["lr"])
        torch.set_grad_enabled(True)

    def __getstate__(self):
        """We modify the state so the object can be pickled"""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["_lock"]
        return state

    def __setstate__(self, state: dict):
        """We modify the state so the object can be pickled"""
        self.__dict__.update(state)
        # Restore the unpickable entries
        self._lock = Lock()

    @property
    def pause_learning(self):
        return self._pause_training

    @pause_learning.setter
    def pause_learning(self, pause: bool):
        self._pause_training = pause

    def change_device(self, device: str):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._proprio_graph.change_device(device)
        self._mission_graph.change_device(device)
        self._feature_extractor.change_device(device)

        self._model = self._model.to(device)
        self._last_trained_model = self._last_trained_model.to(device)

    def update_features(self, node: MissionNode):
        """Extracts visual features from a node that stores an image

        Args:
            node (MissionNode): new node in the mission graph
        """

        # Extract features
        edges, feat, seg, center = self._feature_extractor.extract(img=node.image.clone()[None], return_centers=True)

        # Set features in node
        node.feature_type = self._feature_extractor.get_type()
        node.features = feat
        node.feature_edges = edges
        node.feature_segments = seg
        node.feature_positions = center

    def update_prediction(self, node: MissionNode):
        with self._lock:
            if self._last_trained_model is not None:
                data = Data(x=node.features, edge_index=node.feature_edges)
                node.prediction = self._last_trained_model(data)

    def add_mission_node(self, node: MissionNode):
        """Adds a node to the local graph to images and training info

        Args:
            node (BaseNode): new node in the image graph
        """

        # Compute image features
        self.update_features(node)

        # Add image node
        if self._mission_graph.add_node(node):
            print(f"adding node [{node}]")
            # Project past footprints on current image
            image_projector = node.image_projector
            pose_cam_in_world = node.pose_cam_in_world[None]
            supervision_mask = node.image * 0

            proprio_nodes = self._proprio_graph.get_nodes()
            for pnode, next_pnode in zip(proprio_nodes[:-1], proprio_nodes[1:]):
                side_points = pnode.get_side_points()
                next_side_points = next_pnode.get_side_points()
                points = torch.concat((side_points, next_side_points), dim=0)
                footprint = make_polygon_from_points(points, grid_size=10)[None]
                footprint = pnode.get_footprint_points()[None]
                color = torch.FloatTensor([1.0, 1.0, 1.0]) * pnode.traversability.cpu()  # TODO fix this

                # Project and render mask
                mask, _, _, _ = image_projector.project_and_render(pose_cam_in_world, footprint, color)
                mask = mask[0]

                # Update supervision mask
                supervision_mask = torch.maximum(supervision_mask, mask.to(supervision_mask.device))

            # Finally overwrite the current mask
            node.supervision_mask = supervision_mask
            return True
        else:
            return False

    def add_proprio_node(self, pnode: ProprioceptionNode):
        """Adds a node to the local graph to store proprioception

        Args:
            node (BaseNode): new node in the proprioceptive graph
        """
        if not pnode.is_valid():
            return False

        if not self._proprio_graph.add_node(pnode):
            # Update traversability of latest node
            last_pnode = self._proprio_graph.get_last_node()
            last_pnode.update_traversability(pnode.traversability, pnode.traversability_var)
            return False

        else:
            # Get last added proprio node
            last_pnode = self._proprio_graph.get_last_node()

            # update footprint
            last_side_points = last_pnode.get_side_points()
            side_points = pnode.get_side_points()
            points = torch.concat((last_side_points, side_points), dim=0)
            footprint = make_polygon_from_points(points, grid_size=20)[None]
            footprint = pnode.get_footprint_points()[None]
            color = torch.FloatTensor([1.0, 1.0, 1.0]) * pnode.traversability.cpu()  # TODO fix this

            # Get last mission node
            last_mission_node = self._mission_graph.get_last_node()
            if last_mission_node is None:
                return False

            mission_nodes = self._mission_graph.get_nodes_within_radius_range(
                last_mission_node, 0, self._proprio_graph.max_distance
            )
            # Project footprint onto all the image nodes
            for mnode in mission_nodes:
                # Project
                image_projector = mnode.image_projector
                pose_cam_in_world = mnode.pose_cam_in_world[None]
                supervision_mask = mnode.supervision_mask

                mask, _, _, _ = image_projector.project_and_render(pose_cam_in_world, footprint, color)

                if mask is None or supervision_mask is None:
                    continue

                # Update traversability mask
                mask = mask[0]
                supervision_mask = torch.maximum(supervision_mask, mask.to(supervision_mask.device))

                # Get global node and update supervision signal
                mnode.supervision_mask = supervision_mask
                mnode.update_supervision_signal()

            return True

    def get_mission_nodes(self):
        return self._mission_graph.get_nodes()

    def get_proprio_nodes(self):
        return self._proprio_graph.get_nodes()

    def get_last_valid_mission_node(self):
        last_valid_node = None
        for node in self._mission_graph.get_nodes():
            if node.is_valid():
                last_valid_node = node
        return last_valid_node

    def save(self, mission_path: str, filename: str):
        """Saves a pickled file of the TraversabilityEstimator class

        Args:
            mission_path (str): folder to store the mission
            filename (str): name for the output file
        """
        self._pause_training = True
        os.makedirs(mission_path, exist_ok=True)
        output_file = os.path.join(mission_path, filename)
        self.change_device("cpu")
        self._lock = None
        pickle.dump(self, open(output_file, "wb"))
        self._pause_training = False

    @classmethod
    def load(cls, file_path: str, device="cpu"):
        """Loads pickled file and creates an instance of TraversabilityEstimator,
        loading al the required objects to the given device

        Args:
            file_path (str): Full path of the pickle file
            device (str): Device used to load the torch objects
        """
        # Load pickled object
        obj = pickle.load(open(file_path, "rb"))
        obj.change_device(device)
        return obj

    def save_graph(self, mission_path: str, export_debug: bool = False):
        self._pause_training = True
        # Make folder if it doesn't exist
        os.makedirs(mission_path, exist_ok=True)
        os.makedirs(os.path.join(mission_path, "graph"), exist_ok=True)
        os.makedirs(os.path.join(mission_path, "seg"), exist_ok=True)
        os.makedirs(os.path.join(mission_path, "center"), exist_ok=True)
        os.makedirs(os.path.join(mission_path, "img"), exist_ok=True)

        # Get all the current nodes
        mission_nodes = self._mission_graph.get_nodes()
        i = 0
        for node in mission_nodes:
            if node.is_valid():
                node.save(mission_path, i)
                i += 1
        self._pause_training = False

    def make_batch(self, batch_size: int = 8):
        """Samples a batch from the mission_graph

        Args:
            batch_size (int): Size of the batch
        """
        # Get all the current nodes
        mission_nodes = self._mission_graph.get_n_random_valid_nodes(n=batch_size)
        batch = Batch.from_data_list([x.as_pyg_data() for x in mission_nodes])
        return batch

    def train(self):
        """Runs one step of the training loop
        It samples a batch, and optimizes the model.
        It also updates a copy of the model for inference

        """
        if self._pause_training:
            return

        if self._mission_graph.get_num_valid_nodes() > self.min_samples_for_training:
            # Prepare new batch
            batch = self.make_batch(self._exp_cfg["data_module"]["batch_size"])

            # forward pass
            res = self._model(batch)

            # Compute loss only for valid elements [graph.y_valid]
            # traversability loss
            loss_trav = F.mse_loss(res[:, 0][batch.y_valid], batch.y[batch.y_valid])

            # Reconstruction loss
            nc = 1
            loss_reco = F.mse_loss(res[batch.y_valid][:, nc:], batch.x[batch.y_valid])
            loss = self._exp_cfg["loss"]["trav"] * loss_trav + self._exp_cfg["loss"]["reco"] * loss_reco

            # Backprop
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # Update epochs
            self._epoch += 1

            # Print losses
            if self._epoch % 20 == 0:
                print(f"epoch: {self._epoch} | loss: {loss:5f} | loss_trav: {loss_trav:5f} | loss_reco: {loss_reco:5f}")
            # Update model
            with self._lock:
                self.last_trained_model = self._model

    def plot_mission_node_prediction(self, node: MissionNode):
        return self._visualizer.plot_mission_node_prediction(node)

    def plot_mission_node_training(self, node: MissionNode):
        return self._visualizer.plot_mission_node_training(node)


def run_traversability_estimator():

    t = TraversabilityEstimator()
    t.save("/tmp/te.pickle")
    print("Store pickled")
    t2 = TraversabilityEstimator.load("/tmp/te.pickle")
    print("Loaded pickled")


if __name__ == "__main__":
    run_traversability_estimator()
