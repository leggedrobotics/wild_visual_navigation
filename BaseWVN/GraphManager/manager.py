

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
from pytictac import accumulate_time,ClassContextTimer

from .graphs import (
    BaseGraph,
    DistanceWindowGraph,
    MaxElementsGraph,
)
from .nodes import MainNode,SubNode
from ..utils import ImageProjector,PhyLoss
from ..model import VD_dataset,get_model

to_tensor = transforms.ToTensor()

class Manager:
    def __init__(self,
                device: str = "cuda",
                graph_params = None,
                loss_params = None,
                model_params=None,
                **kwargs):
        self._device = device
        self._label_ext_mode = graph_params.label_ext_mode
        self._vis_node_index = graph_params.vis_node_index
        self._min_samples_for_training = graph_params.min_samples_for_training
        self._update_range_main_graph=graph_params.update_range_main_graph
        self._cut_threshold=graph_params.cut_threshold
        self._edge_dist_thr_main_graph=graph_params.edge_dist_thr_main_graph
        self._extraction_store_folder=graph_params.extraction_store_folder
        self._random_sample_num=graph_params.random_sample_num
        
        self.last_sub_node=None
        
        self._phy_dim=kwargs.get("phy_dim",2)
        self._lr=kwargs.get("lr",0.001)
        

        if self._label_ext_mode:
            self._all_dataset=[]
        self._main_graph = BaseGraph(edge_distance=self._edge_dist_thr_main_graph)
        
        # Visualization node
        self._vis_main_node = None
        self._graph_distance=None
        
        # Mutex
        self._learning_lock = Lock()

        self._pause_training = False
        self._pause_main_graph = False
        self._pause_sub_graph = False
        
        # TODO: self._visualizer = LearningVisualizer()
        #  Init model and optimizer, loss function...
        # Lightning module
        seed_everything(42)
        self._model=get_model(model_params).to(self._device)
        self._model.train()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._phy_loss=PhyLoss(w_pred=loss_params.w_pred,
                               w_reco=loss_params.w_reco,
                               method=loss_params.method,
                               confidence_std_factor=loss_params.confidence_std_factor,
                               log_enabled=loss_params.log_enabled,
                               log_folder=loss_params.log_folder).to(self._device)
        self._loss = torch.tensor([torch.inf])
        self._step = 0

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
        self._main_graph.change_device(device)
        self._model = self._model.to(device)
        self._phy_loss=self._phy_loss.to(device)
    
    
    @accumulate_time
    def update_visualization_node(self):
        # For the first nodes we choose the visualization node as the last node available
        valid_num=self._main_graph.get_num_valid_nodes()
        if valid_num <= self._vis_node_index:
            # self._vis_main_node = self._main_graph.get_nodes()[0]
            if valid_num==0:
                return
            self._vis_main_node=self._main_graph.get_valid_nodes()[0]
        else:

            self._vis_main_node = self._main_graph.get_valid_nodes()[-self._vis_node_index]
        
        # check the head distance between main and sub graph
        last_main_node = self._main_graph.get_last_node()
        last_sub_node = self.last_sub_node
        if last_main_node is not None and last_sub_node is not None:
            self._graph_distance=last_main_node.distance_to(last_sub_node)
    
    @accumulate_time
    def add_main_node(self, node: MainNode,verbose:bool=False,logger=None):
        """ 
        Add new node to the main graph with img and supervision info
        supervision mask has self._phy_dim channels e.g. (2,H,W)
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
                with logger["Lock"]:
                    logger["total main nodes"]=f"{total_nodes}"

            # Init the supervision mask
            H,W=node.image.shape[-2],node.image.shape[-1]
            supervision_mask=torch.ones((self._phy_dim,H,W),dtype=torch.float32,device=self._device)*torch.nan
            node.supervision_mask = supervision_mask
            
            return True
        else:   
            return False
    
    @accumulate_time    
    @torch.no_grad()
    def add_sub_node(self,subnode:SubNode,logger=None):
        if self._pause_sub_graph:
            return False
        if not subnode.is_valid():
            return False
        
        self.last_sub_node=subnode

        feet_planes=subnode.feet_planes
        feet_contact=subnode.feet_contact
        
        last_main_node:MainNode=self._main_graph.get_last_node()
        if last_main_node is None:
            return False
        main_nodes=self._main_graph.get_nodes_within_radius_range(last_main_node,0,self._update_range_main_graph)
        # check if the last main node is too far away from the sub node
        if last_main_node.distance_to(subnode)>self._cut_threshold:
            return False
        num_valid_nodes = self._main_graph.get_num_valid_nodes()
        with logger["Lock"]:
                logger["to_be_updated_mnode_num"]=len(main_nodes)
                logger["num_valid_node"]=num_valid_nodes
        if len(main_nodes)<1:
            return False
       
        # Set color
        color = torch.ones((3,), device=self._device)
        
        C,H,W=last_main_node.supervision_mask.shape
        B=len(main_nodes)
        
        # prepare batches
        K = torch.eye(4, device=self._device).repeat(B, 1, 1)
        supervision_masks=torch.ones((1,C,H,W), device=self._device).repeat(B, 1, 1, 1)
        pose_camera_in_world = torch.eye(4, device=self._device).repeat(B, 1, 1)
        H = last_main_node.image_projector.camera.height
        W = last_main_node.image_projector.camera.width
        
        for i, mnode in enumerate(main_nodes):
            K[i] = mnode.image_projector.camera.intrinsics
            pose_camera_in_world[i] = mnode.pose_cam_in_world
            if mnode.supervision_mask is None:
                print("strange")
                pass
            supervision_masks[i] = mnode.supervision_mask
        with ClassContextTimer(parent_obj=self,block_name="reprojection_main",parent_method_name="add_sub_node"):
            im=ImageProjector(K,H,W)
            assert feet_planes.shape[0]==4
            for i in range(feet_planes.shape[0]):
                # skip not contacting feet
                if not feet_contact[i]:
                    continue
                foot_plane=feet_planes[i].unsqueeze(0)
                foot_plane=foot_plane.repeat(B,1,1)
                with ClassContextTimer(parent_obj=self,block_name="reprojection_main_1",parent_method_name="add_sub_node"):
                    mask, _, _, _ = im.project_and_render(pose_camera_in_world, foot_plane, color)
                print(im.timer)
                mask=mask[:,:self._phy_dim,:,:]*subnode.phy_pred[:,i][None,:,None,None]
                supervision_masks=torch.fmin(supervision_masks,mask)
        # Update supervision mask per node
        for i, mnode in enumerate(main_nodes):
            mnode.supervision_mask = supervision_masks[i]
            mnode.update_supervision_signal()
            with logger["Lock"]:
                logger[f"mnode {i} reproj_pixels_num"]=(~torch.isnan(mnode.supervision_mask[0])).sum().item()
            
        return True

    def get_main_nodes(self):
        return self._main_graph.get_nodes()
    
    def get_last_valid_main_node(self):
        last_valid_node = None
        for node in self._main_graph.get_nodes():
            if node.is_valid():
                last_valid_node = node
        return last_valid_node
    
    def get_main_node_for_visualization(self):
        return self._vis_main_node
    
    @accumulate_time
    def make_batch_to_dataset(
        self,
        node_num: int = 8,
    ):
        # Just sample N random nodes
        mnodes = self._main_graph.get_n_random_valid_nodes(n=node_num)
        with ClassContextTimer(parent_obj=self,block_name="query",parent_method_name="make_batch_to_dataset"):
            batch_list=[mnode.query_valid_batch() for mnode in mnodes]
        with ClassContextTimer(parent_obj=self,block_name="into VDdataset",parent_method_name="make_batch_to_dataset"):
            dataset=VD_dataset(batch_list,combine_batches=True,random_num=self._random_sample_num)
        
        return dataset
      
    def save(self, manager_path: str, filename: str):
        """Saves a pickled file of the Manager class

        Args:
            mission_path (str): folder to store the mission
            filename (str): name for the output file
        """
        self._pause_training = True
        os.makedirs(manager_path, exist_ok=True)
        output_file = os.path.join(manager_path, filename)
        # if not filename.endswith('.pkl') and not filename.endswith('.pickle'):
        #     output_file_graph = output_file+ '.pkl'  # Append .pkl if not already present
        # self.change_device("cpu")
        # self._learning_lock = None
        if not filename.endswith('_data.pt') :
            output_file_datasets = output_file+'_data.pt'
        torch.save(self._all_dataset,output_file_datasets)
        
        if not filename.endswith('_nodes.pt') :
            output_file_graph = output_file+'_nodes.pt'
        torch.save(self._main_graph.get_valid_nodes(),output_file_graph)
        
        
        # pickle.dump(self, open(output_file_graph, "wb"))
        
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
        # obj.change_device(device)
        return obj
    
    def save_ckpt(self, path: str, checkpoint_name: str = "last_checkpoint.pt"):
        """Saves the torch checkpoint and optimization state

        Args:
            path (str): Folder where to put the data
            checkpoint_name (str): Name for the checkpoint file
        """
        with self._learning_lock:
            self._pause_training = True
            
            # Prepare folder
            os.makedirs(path, exist_ok=True)
            checkpoint_file = os.path.join(path, checkpoint_name)

            # Save checkpoint
            torch.save(
                {
                    "step": self._step,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "phy_loss_state_dict": self._phy_loss.state_dict(),
                    "loss": self._loss.item(),
                },
                checkpoint_file,
            )

            print(f"Saved checkpoint to file {checkpoint_file}")
            self._pause_training = False
    
    def load_ckpt(self, checkpoint_path: str):
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
            self._phy_loss.load_state_dict(checkpoint["phy_loss_state_dict"])
            self._step = checkpoint["step"]
            self._loss = checkpoint["loss"]

            # Set model in training mode
            self._model.train()
            self._optimizer.zero_grad()

            print(f"Loaded checkpoint from file {checkpoint_path}")
            self._pause_training = False
    
    @accumulate_time
    def train(self):
        """Runs one step of the training loop
        It samples a batch, and optimizes the model.
        It also updates a copy of the model for inference

        """
        if self._pause_training:
            return {}
        
        num_valid_nodes = self._main_graph.get_num_valid_nodes()
        return_dict = {"main_graph_num_valid_node": num_valid_nodes}
        if num_valid_nodes > self._min_samples_for_training:
            # Prepare new batch
            dataset=self.make_batch_to_dataset(self._min_samples_for_training)
            if self._label_ext_mode:
                self._all_dataset.append(dataset)
            with self._learning_lock:
                for batch_idx in range(dataset.get_batch_num()):     
                    # Forward pass
                    res = self._model(dataset.get_x(batch_idx))
                    
                    log_step = (self._step % 20) == 0
                    self._loss,confidence,loss_dict = self._phy_loss(dataset, res, step=self._step, log_step=log_step,batch_idx=batch_idx)
                    
                    # Backprop
                    self._optimizer.zero_grad()
                    self._loss.backward()
                    self._optimizer.step()
            # Print losses
            loss_reco=loss_dict["loss_reco"]
            loss_pred=loss_dict["loss_pred"]
            if log_step: 
                print(f"step: {self._step}, loss: {self._loss}, loss_reco: {loss_reco}, loss_pred: {loss_pred}")
            
            # Update steps
            self._step += 1
            
            return_dict["total_loss"] = self._loss.item()
            return_dict["confidence"] = confidence.mean().item()
            return_dict["loss_reco"] = loss_reco.item()
            return_dict["loss_pred"] = loss_pred.item()
            
            return return_dict
        else:
            return_dict["total_loss"] = -1
            return return_dict
            
        