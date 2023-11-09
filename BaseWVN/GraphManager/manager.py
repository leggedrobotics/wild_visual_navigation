

from BaseWVN import WVN_ROOT_DIR
from pytorch_lightning import seed_everything
from torch_geometric.data import Data, Batch
from threading import Lock
import dataclasses
import os
import pickle
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

from .graphs import (
    BaseGraph,
    DistanceWindowGraph,
    MaxElementsGraph,
)
from .nodes import MainNode


to_tensor = transforms.ToTensor()

class Manager:
    def __init__(self,
                device: str = "cuda",
                max_dist_sub_graph: float = 3,
                edge_dist_thr_sub_graph: float = 0.2,
                edge_dist_thr_main_graph: float = 1,
                min_samples_for_training: int = 10,
                vis_node_index: int = 10,
                label_ext_mode: bool = False,
                **kwargs):
        self._device = device
        self._label_ext_mode = label_ext_mode
        self._vis_node_index = vis_node_index
        self._min_samples_for_training = min_samples_for_training
        self._extraction_store_folder=kwargs.get("extraction_store_folder",'LabelExtraction')
        
        # Init main and sub graphs
        self._sub_graph=DistanceWindowGraph(max_distance=max_dist_sub_graph,edge_distance=edge_dist_thr_sub_graph)
        if label_ext_mode:
            self._main_graph = MaxElementsGraph(edge_distance=edge_dist_thr_main_graph, max_elements=200)
        else:
            self._main_graph = BaseGraph(edge_distance=edge_dist_thr_main_graph)
        
        # Visualization node
        self._vis_main_node = None
        
        # Mutex
        self._learning_lock = Lock()

        self._pause_training = False
        self._pause_main_graph = False
        self._pause_sub_graph = False
        
        # TODO: self._visualizer = LearningVisualizer()
        #  Init model and optimizer, loss function...
    
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
        self._sub_graph.change_device(device)
        self._main_graph.change_device(device)
        self._model = self._model.to(device)
    
    def update_prediction(self, node: MainNode):
        # TODO:use MLP to predict here, update_node_confidence
        pass
    
    def update_visualization_node(self):
        # For the first nodes we choose the visualization node as the last node available
        if self._main_graph.get_num_nodes() <= self._vis_node_index:
            self._vis_main_node = self._main_graph.get_nodes()[0]
        else:

            self._vis_main_node = self._main_graph.get_nodes()[-self._vis_node_index]
    
    def add_main_node(self, node: MainNode,verbose:bool=False,logger=None):
        """ 
        Add new node to the main graph with img and supervision info
        supervision mask has 2 channels (2,H,W)
        """
        if self._pause_main_graph:
            return False
        success=self._main_graph.add_node(node)
        if success and node.use_for_training:
            # Print some info
            total_nodes = self._main_graph.get_num_nodes()
            if logger is None:
                s = f"adding node [{node}], "
                s += " " * (48 - len(s)) + f"total nodes [{total_nodes}]"
                if verbose:
                    print(s)
            else:
                logger["total main nodes"]=f"{total_nodes}"

            # Init the supervision mask
            H,W=node.image.shape[-2],node.image.shape[-1]
            supervision_mask=torch.ones((2,H,W),dtype=torch.float32,device=self._device)*torch.nan
            node.supervision_mask = supervision_mask
            
            # TODO: in extract label mode, save the node.image maybe
            
            return True
        else:   
            return False
        
    @torch.no_grad()
    def add_supervision_node(self):
        # TODO: add supervision node to sub_graph
        pass
    
    def get_main_nodes(self):
        return self._main_graph.get_nodes()
    
    def get_supervision_nodes(self):
        return self._sub_graph.get_nodes()
    
    def get_last_valid_main_node(self):
        last_valid_node = None
        for node in self._main_graph.get_nodes():
            if node.is_valid():
                last_valid_node = node
        return last_valid_node
    
    def get_main_node_for_visualization(self):
        return self._vis_main_node
    
    def save(self, manager_path: str, filename: str):
        """Saves a pickled file of the Manager class

        Args:
            mission_path (str): folder to store the mission
            filename (str): name for the output file
        """
        self._pause_training = True
        os.makedirs(manager_path, exist_ok=True)
        output_file = os.path.join(manager_path, filename)
        self.change_device("cpu")
        self._learning_lock = None
        pickle.dump(self, open(output_file, "wb"))
        self._pause_training = False
    
    @classmethod
    def load(cls, file_path: str, device="cpu"):
        """Loads pickled file and creates an instance of Manager,
        loading al the required objects to the given device

        Args:
            file_path (str): Full path of the pickle file
            device (str): Device used to load the torch objects
        """
        # Load pickled object
        obj = pickle.load(open(file_path, "rb"))
        obj.change_device(device)
        return obj
    
    

        
        