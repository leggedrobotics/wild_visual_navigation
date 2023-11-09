

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

from BaseWVN.GraphManager import (
    BaseGraph,
    DistanceWindowGraph,
    VisualNode,
    MaxElementsGraph,
)

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
        self._vis_mission_node = None
        
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
    
    def update_prediction(self, node: VisualNode):
        # TODO:use MLP to predict here, update_node_confidence
        pass
    
    def update_visualization_node(self):
        # For the first nodes we choose the visualization node as the last node available
        if self._main_graph.get_num_nodes() <= self._vis_node_index:
            self._vis_mission_node = self._main_graph.get_nodes()[0]
        else:
            self._vis_mission_node = self._main_graph.get_nodes()[-self._vis_node_index]
    
    def add_visual_node(self, node: VisualNode,verbose:bool=False):
        """ 
        Add new node to the main graph with img and supervision info
        supervision mask has 2 channels (2,H,W)
        """
        if self._pause_mission_graph:
            return False
        success=self._main_graph.add_node(node)
        if success and node.use_for_training:
            # Print some info
            total_nodes = self._main_graph.get_num_nodes()
            s = f"adding node [{node}], "
            s += " " * (48 - len(s)) + f"total nodes [{total_nodes}]"
            if verbose:
                print(s)

            # Init the supervision mask
            H,W=node.img.shape[-2],node.img.shape[-1]
            supervision_mask=torch.ones((2,H,W),dtype=torch.float32,device=self._device)*torch.nan
            node.supervision_mask = supervision_mask
            
            # TODO: in extract label mode, save the node.image maybe
            
            return True
        else:   
            return False

    
    

        
        