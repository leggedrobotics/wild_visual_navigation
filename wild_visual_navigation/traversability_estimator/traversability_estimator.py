from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.learning.model import get_model
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.traversability_estimator import (
    BaseGraph,
    DistanceWindowGraph,
    MissionNode,
    ProprioceptionNode,
)

from wild_visual_navigation.learning.utils import compute_loss
from wild_visual_navigation.utils import make_polygon_from_points
from wild_visual_navigation.visu import LearningVisualizer
from wild_visual_navigation.utils import KLTTracker, KLTTrackerOpenCV
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
        optical_flow_estimatior_type: str = "none",
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
        # Optical flow
        self._optical_flow_estimatior_type = optical_flow_estimatior_type

        if optical_flow_estimatior_type == "sparse":
            self._optical_flow_estimatior = KLTTrackerOpenCV(device=device)

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
        self._loss = torch.tensor([torch.inf])
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
    def loss(self):
        return self._loss.detach().item()

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
        if self._optical_flow_estimatior_type != "none":
            self._optical_flow_estimatior = self._optical_flow_estimatior.to(device)

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
                with torch.inference_mode():
                    node.prediction = self._last_trained_model(data)

    def add_corrospondences_dense_optical_flow(self, node: MissionNode, previous_node: MissionNode, debug=False):
        flow_previous_to_current = self._optical_flow_estimatior.forward(
            previous_node.image.clone(), node.image.clone()
        )
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, previous_node.image.shape[1], device=self.device, dtype=torch.float32),
            torch.arange(0, previous_node.image.shape[2], device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        positions = torch.stack([grid_x, grid_y])
        positions += flow_previous_to_current
        positions = positions.type(torch.long)
        start_seg = torch.unique(previous_node.feature_segments)

        previous = []
        current = []
        for el in start_seg:
            m = previous_node.feature_segments == el
            candidates = positions[:, m]
            m = (
                (candidates[0, :] >= 0)
                * (candidates[0, :] < m.shape[0])
                * (candidates[1, :] >= 0)
                * (candidates[1, :] < m.shape[1])
            )
            if m.sum() == 0:
                continue
            candidates = candidates[:, m]
            res = node.feature_segments[candidates[0, :], candidates[1, :]]
            previous.append(el)
            current.append(torch.unique(res, sorted=True)[0])

        if len(current) != 0:
            current = torch.stack(current)
            previous = torch.stack(previous)
            corrospondence = torch.stack([previous, current], dim=1)
            node.corrospondence = corrospondence

            if debug:
                from wild_visual_navigation import WVN_ROOT_DIR
                from wild_visual_navigation.visu import LearningVisualizer

                visu = LearningVisualizer(p_visu=os.path.join(WVN_ROOT_DIR, "results/test_visu"), store=True)
                visu.plot_corrospondence_segment(
                    seg_prev=previous_node.feature_segments,
                    seg_current=node.feature_segments,
                    img_prev=previous_node.image,
                    img_current=node.image,
                    center_prev=previous_node.feature_positions,
                    center_current=node.feature_positions,
                    corrospondence=node.corrospondence,
                    tag="centers",
                )

                visu.plot_optical_flow(
                    flow=flow_previous_to_current, img1=previous_node.image, img2=node.image, tag="flow", s=50
                )

    def add_corrospondences_sparse_optical_flow(self, node: MissionNode, previous_node: MissionNode, debug=False):
        # Transform previous_nodes feature locations into current image using KLT
        pre_pos = previous_node.feature_positions
        cur_pos = self._optical_flow_estimatior(
            previous_node.feature_positions[:, 0].clone(),
            previous_node.feature_positions[:, 1].clone(),
            previous_node.image,
            node.image,
        )
        cur_pos = torch.stack(cur_pos, dim=1)
        # Use forward propagated cluster centers to get segment index of current image

        # Only compute for forward propagated centers that fall onto current image plane
        m = (
            (cur_pos[:, 0] >= 0)
            * (cur_pos[:, 1] >= 0)
            * (cur_pos[:, 0] < node.image.shape[1])
            * (cur_pos[:, 1] < node.image.shape[2])
        )

        # Can enumerate previous segments and use mask to get segment index
        pre = torch.arange(previous_node.feature_positions.shape[0], device=self.device)[m]
        cor_pre = pre[m]

        # Check feature_segmentation mask to index correct segment index
        cur_pos = cur_pos[m].type(torch.long)
        cor_cur = node.feature_segments[cur_pos[:, 1], cur_pos[:, 0]]
        node.corrospondence = torch.stack([cor_pre, cor_cur], dim=1)

        if debug:
            from wild_visual_navigation import WVN_ROOT_DIR
            from wild_visual_navigation.visu import LearningVisualizer

            visu = LearningVisualizer(p_visu=os.path.join(WVN_ROOT_DIR, "results/test_visu"), store=True)
            visu.plot_sparse_optical_flow(
                pre_pos,
                cur_pos,
                img1=previous_node.image,
                img2=node.image,
                tag="flow",
            )
            visu.plot_corrospondence_segment(
                seg_prev=previous_node.feature_segments,
                seg_current=node.feature_segments,
                img_prev=previous_node.image,
                img_current=node.image,
                center_prev=previous_node.feature_positions,
                center_current=node.feature_positions,
                corrospondence=node.corrospondence,
                tag="centers",
            )

    def add_mission_node(self, node: MissionNode):
        """Adds a node to the local graph to images and training info

        Args:
            node (BaseNode): new node in the image graph
        """

        # Compute image features
        self.update_features(node)

        previous_node = self._mission_graph.get_last_node()
        # Add image node
        if self._mission_graph.add_node(node):
            total_nodes = self._mission_graph.get_num_nodes()
            s = f"adding node [{node}], "
            s += " " * (48 - len(s)) + f"total nodes [{total_nodes}]"
            print(s)
            # Set optical flow
            if self._optical_flow_estimatior_type == "dense" and previous_node is not None:
                raise Exception("Not working")
                self.add_corrospondences_dense_optical_flow(node, previous_node, debug=False)

            elif self._optical_flow_estimatior_type == "sparse" and previous_node is not None:
                self.add_corrospondences_sparse_optical_flow(node, previous_node, debug=False)

            # Project past footprints on current image
            image_projector = node.image_projector
            pose_cam_in_world = node.pose_cam_in_world[None]
            supervision_mask = torch.ones(node.image.shape).to(node.image.device) * torch.nan

            proprio_nodes = self._proprio_graph.get_nodes()
            for pnode, next_pnode in zip(proprio_nodes[:-1], proprio_nodes[1:]):
                side_points = pnode.get_side_points()
                next_side_points = next_pnode.get_side_points()
                points = torch.concat((side_points, next_side_points), dim=0)
                footprint = make_polygon_from_points(points, grid_size=10)[None]
                # footprint = pnode.get_footprint_points()[None]
                color = torch.FloatTensor([1.0, 1.0, 1.0]) * pnode.traversability.cpu()  # TODO fix this

                # Project and render mask
                mask, _, _, _ = image_projector.project_and_render(pose_cam_in_world, footprint, color)
                mask = mask[0]

                # Update supervision mask
                supervision_mask = torch.fmax(supervision_mask, mask.to(supervision_mask.device))

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
        # If the node is not valid, we do nothing
        if not pnode.is_valid():
            return False

        # Get last added proprio node
        last_pnode = self._proprio_graph.get_last_node()

        if not self._proprio_graph.add_node(pnode):
            # Update traversability of latest node
            if last_pnode is not None:
                last_pnode.update_traversability(pnode.traversability, pnode.traversability_var)
            return False

        else:
            # If the previous node doesn't exist or is invalid, we do nothing
            if last_pnode is None or not last_pnode.is_valid():
                return False

            # update footprint
            last_side_points = last_pnode.get_side_points()
            side_points = pnode.get_side_points()
            points = torch.concat((last_side_points, side_points), dim=0)
            footprint = make_polygon_from_points(points, grid_size=20)[None]
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
                supervision_mask = torch.fmax(supervision_mask, mask.to(supervision_mask.device))

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
                node.save(mission_path, i, graph_only=False, previous_node=self._mission_graph.get_previous_node(node))
                i += 1
        self._pause_training = False

    def save_checkpoint(self, mission_path: str, filename: str = "last_model.pt"):
        """Saves the torch model and optimization state

        Args:
            mission_path (str): Folder where to put the data
            filename (str): Name for the checkpoint file
        """

        self._pause_training = True
        # Prepare folder
        os.makedirs(mission_path, exist_ok=True)
        checkpoint_file = os.path.join(mission_path, filename)

        # Save checkpoint
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": self._loss,
            },
            checkpoint_file,
        )

        print(f"Saved checkpoint to file {checkpoint_file}")
        self._pause_training = False

    def load_checkpoint(self, mission_path: str, filename: str = "last_model.pt"):
        """Loads the torch model and optimization state

        Args:
            mission_path (str): Folder where to put the data
            checkpoint_file (str): Name of the checkpoint file to be loaded
        """

        self._pause_training = True

        # Prepare file
        os.makedirs(mission_path, exist_ok=True)
        checkpoint_file = os.path.join(mission_path, filename)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"]
        self._loss = checkpoint["loss"]

        # Copy last trained model
        with torch.inference_mode():
            self._last_trained_model = self._model

        # Set model in training mode
        self._model.train()

        print(f"Loaded checkpoint from file {checkpoint_file}")
        self._pause_training = False

    def make_batch(self, batch_size: int = 8):
        """Samples a batch from the mission_graph

        Args:
            batch_size (int): Size of the batch
        """
        # Get all the current nodes

        if self._optical_flow_estimatior_type != "none":
            mission_nodes = self._mission_graph.get_n_random_valid_nodes(n=batch_size)
            ls = [
                x.as_pyg_data(self._mission_graph.get_previous_node(x))
                for x in mission_nodes
                if x.corrospondence is not None
            ]

            ls_aux = [self._mission_graph.get_previous_node(x).as_pyg_data(aux=True) for x in mission_nodes]

            # Make sure to only use nodes with valid corrospondences
            while len(ls) < batch_size:
                mn = self._mission_graph.get_n_random_valid_nodes(n=1)[0]
                if mn.corrospondence is not None:
                    ls.append(mn.as_pyg_data(self._mission_graph.get_previous_node(mn)))
                    ls_aux.append(self._mission_graph.get_previous_node(mn).as_pyg_data(aux=True))

            batch = [Batch.from_data_list(ls), Batch.from_data_list(ls_aux)]
        else:
            mission_nodes = self._mission_graph.get_n_random_valid_nodes(n=batch_size)
            batch = [Batch.from_data_list([x.as_pyg_data() for x in mission_nodes]), None]

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
            graph, graph_aux = self.make_batch(self._exp_cfg["data_module"]["batch_size"])

            # forward pass
            res = self._model(graph)
            self._loss, loss_aux = compute_loss(graph, res, self._exp_cfg["loss"], self._model, graph_aux)

            # Backprop
            self._optimizer.zero_grad()
            self._loss.backward()
            self._optimizer.step()

            # Update epochs
            self._epoch += 1

            # Print losses
            if self._epoch % 20 == 0:
                loss_trav = loss_aux["loss_trav"]
                loss_reco = loss_aux["loss_reco"]
                print(
                    f"epoch: {self._epoch} | loss: {self._loss:5f} | loss_trav: {loss_trav:5f} | loss_reco: {loss_reco:5f}"
                )
            # Update model
            with self._lock:
                self.last_trained_model = self._model

            # Return loss
            return self._loss.item()

    def plot_mission_node_prediction(self, node: MissionNode):
        return self._visualizer.plot_mission_node_prediction(node)

    def plot_mission_node_training(self, node: MissionNode):
        return self._visualizer.plot_mission_node_training(node)


def run_traversability_estimator():
    t = TraversabilityEstimator()
    t.save("/tmp", "te.pickle")
    print("Store pickled")
    t2 = TraversabilityEstimator.load("/tmp/te.pickle")
    print("Loaded pickled")


if __name__ == "__main__":
    run_traversability_estimator()
