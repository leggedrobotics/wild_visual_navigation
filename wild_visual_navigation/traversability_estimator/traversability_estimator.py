#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.model import get_model
from wild_visual_navigation.cfg import ExperimentParams
from pytictac import accumulate_time
from wild_visual_navigation.traversability_estimator import (
    BaseGraph,
    DistanceWindowGraph,
    MissionNode,
    SupervisionNode,
    MaxElementsGraph,
)
from wild_visual_navigation.utils import WVNMode
from wild_visual_navigation.utils import TraversabilityLoss, AnomalyLoss
from wild_visual_navigation.visu import LearningVisualizer

from pytorch_lightning import seed_everything
from wild_visual_navigation.utils import Data, Batch
from threading import Lock
import os
import pickle
import torch
import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()


class TraversabilityEstimator:
    def __init__(
        self,
        params: ExperimentParams,
        device: str,
        max_distance: float,
        image_distance_thr: float,
        supervision_distance_thr: float,
        min_samples_for_training: int,
        vis_node_index: int,
        mode: bool,
        extraction_store_folder,
        anomaly_detection: bool,
    ):
        self._device = device
        self._mode = mode
        self._extraction_store_folder = extraction_store_folder
        self._min_samples_for_training = min_samples_for_training
        self._vis_node_index = vis_node_index
        self._params = params
        self._anomaly_detection = anomaly_detection

        # Local graphs
        self._supervision_graph = DistanceWindowGraph(max_distance=max_distance, edge_distance=supervision_distance_thr)

        # Experience graph
        if mode == WVNMode.EXTRACT_LABELS:
            self._mission_graph = MaxElementsGraph(edge_distance=image_distance_thr, max_elements=200)
        else:
            self._mission_graph = BaseGraph(edge_distance=image_distance_thr)

        # Visualization node
        self._vis_mission_node = None

        # Mutex
        self._learning_lock = Lock()

        self._pause_training = False
        self._pause_mission_graph = False
        self._pause_supervision_graph = False

        # Visualization
        self._visualizer = LearningVisualizer()

        # Lightning module
        seed_everything(42)

        self._model = get_model(self._params.model).to(self._device)
        self._model.train()

        if self._anomaly_detection:
            self._traversability_loss = AnomalyLoss(
                **self._params["loss_anomaly"],
                log_enabled=self._params["general"]["log_confidence"],
                log_folder=self._params["general"]["model_path"],
            )
            self._traversability_loss.to(self._device)

        else:
            self._traversability_loss = TraversabilityLoss(
                **self._params["loss"],
                model=self._model,
                log_enabled=self._params["general"]["log_confidence"],
                log_folder=self._params["general"]["model_path"],
            )
            self._traversability_loss.to(self._device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._params["optimizer"]["lr"])
        self._loss = torch.tensor([torch.inf])
        self._step = 0
        self._debug_info_node_count = 0

        torch.set_grad_enabled(True)

    def __getstate__(self):
        """We modify the state so the object can be pickled"""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["_learning_lock"]
        return state

    def __setstate__(self, state: dict):
        """We modify the state so the object can be pickled"""
        self.__dict__.update(state)
        # Restore the unpickable entries
        self._learning_lock = Lock()

    def reset(self):
        print("[WARNING] Resetting the traversability estimator is not fully tested")

    @property
    def loss(self):
        return self._loss.detach().item()

    @property
    def step(self):
        return self._step

    @property
    def pause_learning(self):
        return self._pause_training

    @pause_learning.setter
    def pause_learning(self, pause: bool):
        self._pause_training = pause

    @accumulate_time
    def change_device(self, device: str):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._supervision_graph.change_device(device)
        self._mission_graph.change_device(device)
        self._model = self._model.to(device)

        if self._use_feature_extractor:
            self._feature_extractor.change_device(device)

    @accumulate_time
    def update_visualization_node(self):
        # For the first nodes we choose the visualization node as the last node available
        if self._mission_graph.get_num_nodes() <= self._vis_node_index:
            self._vis_mission_node = self._mission_graph.get_nodes()[0]
        else:
            # We remove debug data if we are in online mode (after optical flow, so the image is still available)
            if self._mode == WVNMode.ONLINE and self._vis_mission_node is not None:
                self._vis_mission_node.clear_debug_data()

            self._vis_mission_node = self._mission_graph.get_nodes()[-self._vis_node_index]

    @accumulate_time
    def add_mission_node(self, node: MissionNode, verbose: bool = False):
        """Adds a node to the mission graph to images and training info

        Args:
            node (BaseNode): new node in the image graph
        """

        if self._pause_mission_graph:
            return False

        # Add image node
        success = self._mission_graph.add_node(node)

        if success and node.use_for_training:
            # Print some info
            total_nodes = self._mission_graph.get_num_nodes()
            s = f"adding node [{node}], "
            s += " " * (48 - len(s)) + f"total nodes [{total_nodes}]"
            if verbose:
                print(s)
            h, w = node._feature_segments.shape[0], node._feature_segments.shape[1]
            # Project past footprints on current image
            supervision_mask = torch.ones((3, h, w)).to(self._device) * torch.nan

            # Finally overwrite the current mask
            node.supervision_mask = supervision_mask
            node.update_supervision_signal()

            return True

        return False

    @accumulate_time
    @torch.no_grad()
    def add_supervision_node(self, pnode: SupervisionNode):
        """Adds a node to the supervision graph to store supervision

        Args:
            node (BaseNode): new node in the supervision graph
        """
        if self._pause_supervision_graph:
            return False

        # If the node is not valid, we do nothing
        if not pnode.is_valid():
            return False

        # Get last added supervision node
        last_pnode = self._supervision_graph.get_last_node()
        success = self._supervision_graph.add_node(pnode)

        if not success:
            # Update traversability of latest node
            if last_pnode is not None:
                last_pnode.update_traversability(pnode.traversability, pnode.traversability_var)
            return False

        else:

            # If the previous node doesn't exist or it's invalid, we do nothing
            if last_pnode is None or not last_pnode.is_valid():
                return False

            # Update footprint
            footprint = pnode.make_footprint_with_node(last_pnode)[None]

            # Get last mission node
            last_mission_node = self._mission_graph.get_last_node()
            if last_mission_node is None:
                return False
            if (not hasattr(last_mission_node, "supervision_mask")) or (last_mission_node.supervision_mask is None):
                return False

            for j, ele in enumerate(
                list(self._mission_graph._graph.nodes._nodes.items())[self._debug_info_node_count :]
            ):
                node, values = ele
                if last_mission_node.timestamp - values["timestamp"] > 30:
                    node.clear_debug_data()
                    self._debug_info_node_count += 1
                else:
                    break

            # Get all mission nodes within a range
            mission_nodes = self._mission_graph.get_nodes_within_radius_range(
                last_mission_node, 0, self._supervision_graph.max_distance
            )

            if len(mission_nodes) < 1:
                return False

            # Set color
            color = torch.ones((3,), device=self._device)

            # New implementation
            B = len(mission_nodes)
            # Prepare batches
            K = torch.eye(4, device=self._device).repeat(B, 1, 1)
            supervision_masks = torch.zeros(last_mission_node.supervision_mask.shape, device=self._device).repeat(
                B, 1, 1, 1
            )
            pose_camera_in_world = torch.eye(4, device=self._device).repeat(B, 1, 1)
            H = last_mission_node.image_projector.camera.height
            W = last_mission_node.image_projector.camera.width
            footprints = footprint.repeat(B, 1, 1)

            for i, mnode in enumerate(mission_nodes):
                K[i] = mnode.image_projector.camera.intrinsics
                pose_camera_in_world[i] = mnode.pose_cam_in_world

                if not ((not hasattr(mnode, "supervision_mask")) or (mnode.supervision_mask is None)):
                    supervision_masks[i] = mnode.supervision_mask

            im = ImageProjector(K, H, W)
            mask, _, _, _ = im.project_and_render(pose_camera_in_world, footprints, color)

            # Update traversability
            mask = mask * pnode.traversability
            supervision_masks = torch.fmin(supervision_masks, mask)

            # Update supervision mask per node
            for i, mnode in enumerate(mission_nodes):
                mnode.supervision_mask = supervision_masks[i]
                mnode.update_supervision_signal()

                if self._mode == WVNMode.EXTRACT_LABELS:
                    p = os.path.join(
                        self._extraction_store_folder,
                        "supervision_mask",
                        str(mnode.timestamp).replace(".", "_") + ".pt",
                    )
                    store = torch.nan_to_num(mnode.supervision_mask.nanmean(axis=0)) != 0
                    torch.save(store, p)

            return True

    def get_mission_nodes(self):
        return self._mission_graph.get_nodes()

    def get_supervision_nodes(self):
        return self._supervision_graph.get_nodes()

    def get_last_valid_mission_node(self):
        last_valid_node = None
        for node in self._mission_graph.get_nodes():
            if node.is_valid():
                last_valid_node = node
        return last_valid_node

    def get_mission_node_for_visualization(self):
        return self._vis_mission_node

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
        self._learning_lock = None
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
        """Saves the graph as a dataset for offline training

        Args:
            mission_path (str): Folder where to put the data
            export_debug (bool): If debug data should be exported as well (e.g. images)
        """

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
                node.save(
                    mission_path,
                    i,
                    graph_only=False,
                    previous_node=self._mission_graph.get_previous_node(node),
                )
                i += 1
        self._pause_training = False

    def save_checkpoint(self, mission_path: str, checkpoint_name: str = "last_checkpoint.pt"):
        """Saves the torch checkpoint and optimization state

        Args:
            mission_path (str): Folder where to put the data
            checkpoint_name (str): Name for the checkpoint file
        """
        with self._learning_lock:
            self._pause_training = True

            # Prepare folder
            os.makedirs(mission_path, exist_ok=True)
            checkpoint_file = os.path.join(mission_path, checkpoint_name)

            # Save checkpoint
            torch.save(
                {
                    "step": self._step,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "traversability_loss_state_dict": self._traversability_loss.state_dict(),
                    "loss": self._loss.item(),
                },
                checkpoint_file,
            )

            print(f"Saved checkpoint to file {checkpoint_file}")
            self._pause_training = False

    def load_checkpoint(self, checkpoint_path: str):
        """Loads the torch checkpoint and optimization state

        Args:
            checkpoint_path (str): Global path to the checkpoint
        """

        with self._learning_lock:
            self._pause_training = True

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._traversability_loss.load_state_dict(checkpoint["traversability_loss_state_dict"])
            self._step = checkpoint["step"]
            self._loss = checkpoint["loss"]

            # Set model in training mode
            self._model.train()
            self._optimizer.zero_grad()

            print(f"Loaded checkpoint from file {checkpoint_path}")
            self._pause_training = False

    @accumulate_time
    def make_batch(
        self,
        batch_size: int = 8,
    ):
        """Samples a batch from the mission_graph

        Args:
            batch_size (int): Size of the batch
        """

        # Just sample N random nodes
        mission_nodes = self._mission_graph.get_n_random_valid_nodes(n=batch_size)
        batch = Batch.from_data_list([x.as_pyg_data(anomaly_detection=self._anomaly_detection) for x in mission_nodes])

        return batch

    @accumulate_time
    def train(self):
        """Runs one step of the training loop
        It samples a batch, and optimizes the model.
        It also updates a copy of the model for inference

        """
        if self._pause_training:
            return {}

        num_valid_nodes = self._mission_graph.get_num_valid_nodes()
        return_dict = {"mission_graph_num_valid_node": num_valid_nodes}
        if num_valid_nodes > self._min_samples_for_training:
            # Prepare new batch
            graph = self.make_batch(self._params["ablation_data_module"]["batch_size"])
            if graph is not None:
                with self._learning_lock:
                    # Forward pass

                    res = self._model(graph)

                    log_step = (self._step % 20) == 0
                    self._loss, loss_aux, trav = self._traversability_loss(
                        graph, res, step=self._step, log_step=log_step
                    )

                    # Backprop
                    self._optimizer.zero_grad()
                    self._loss.backward()
                    self._optimizer.step()

                # Print losses
                if log_step:
                    loss_trav = loss_aux["loss_trav"]
                    loss_reco = loss_aux["loss_reco"]
                    print(
                        f"step: {self._step} | loss: {self._loss.item():5f} | loss_trav: {loss_trav.item():5f} | loss_reco: {loss_reco.item():5f}"
                    )

                # Update steps
                self._step += 1

                # Return loss
                return_dict["loss_total"] = self._loss.item()
                return_dict["loss_trav"] = loss_aux["loss_trav"].item()
                return_dict["loss_reco"] = loss_aux["loss_reco"].item()

                return return_dict
        return_dict["loss_total"] = -1
        return return_dict

    @accumulate_time
    def plot_mission_node_prediction(self, node: MissionNode):
        return self._visualizer.plot_mission_node_prediction(node)

    @accumulate_time
    def plot_mission_node_training(self, node: MissionNode):
        return self._visualizer.plot_mission_node_training(node)
