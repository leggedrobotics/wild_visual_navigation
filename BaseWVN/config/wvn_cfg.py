""" 
This file contains the all configurations for the Wild-Visual-Navigation project.
 """
from dataclasses import dataclass, field,asdict,is_dataclass
from typing import Tuple, Dict, List, Optional
from simple_parsing.helpers import Serializable
import numpy as np
import yaml
@dataclass
class ParamCollection(Serializable):
    """A collection of parameters."""
    @dataclass
    class GeneralParams:
        """General parameters for the experiment."""
        name: str='debug/debug'
        timestamp: bool=True
        model_path: str='model'
        # ... [rest of the attributes]
    general: GeneralParams=GeneralParams()

    @dataclass
    class RosParams:
        """Parameters for ROS."""

        anymal_bag_name: str='lpc'
        anymal_state_topic: str='/state_estimator/anymal_state'
        feet_list: List[str]=field(default_factory=lambda: ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'])
        phy_decoder_input_topic: str='/debug_info'
        phy_decoder_output_topic:str='/vd_pipeline/phy_decoder_out'

        camera_bag_name: str='jetson'
        camera_topic: str='/v4l2_camera/image_raw_throttle/compressed'
        camera_info_topic: str='/v4l2_camera/camera_info_throttle'

        fixed_frame: str='odom'
        base_frame: str='base'
        footprint_frame: str='footprint'

        robot_length: float=0.930
        robot_height: float=0.890
        robot_width: float=0.530
        robot_max_velocity: float=1.2
        foot_radius: float=0.03269
        rear_camera_in_base= np.array([[ 3.63509049e-06, -1.43680305e-01, -9.89624138e-01,-3.53700000e-01],
                                        [-9.99999820e-01,  1.34923159e-11, -3.67320444e-06,0.00000000e+00],
                                        [ 5.27780582e-07,  9.89623958e-01, -1.43680305e-01,1.63400000e-01],
                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]])
        pass
    roscfg: RosParams=RosParams()
    
    @dataclass
    class ThreadParams:
        """Parameters for the threads."""
        image_callback_rate: float=1.0
        proprio_callback_rate: float=2.0
        learning_rate: float=0.5
        logging_rate: float=0.5
    
    thread: ThreadParams=ThreadParams()

    @dataclass
    class RunParams:
        """Parameters for the run."""
        device: str='cuda'
        mode: str='debug'
        palette: str='husl'
        print_time: bool=True
        pass
    run: RunParams=RunParams()
    
    @dataclass
    class LoggerParams:
        name: str = "neptune"
        neptune_project_name: str = "RSL/WVN"

    logger: LoggerParams = LoggerParams()

    @dataclass
    class OptimizerParams:
        name: str = "ADAM"
        lr: float = 0.001

    optimizer: OptimizerParams = OptimizerParams()

    @dataclass
    class FeatParams:
        """Parameters for the feature extractor."""
        segmentation_type: str='pixel'
        feature_type: str='dinov2'
        input_size:int =1260
        interp: str='bilinear'
        center_crop: bool=False
        physical_dim:int=2
    
    feat: FeatParams=FeatParams()

    @dataclass
    class LossParams:
        w_pred: float = 0.1
        w_reco: float = 0.9
        method: str = "running_mean"
        confidence_std_factor: float = 1.0
        confidence_threshold: float = 0.5
        confidence_mode: str = "gmm_1d" # gmm_1d,gmm_all,fixed
        log_enabled: bool = False
        log_folder: str = "/tmp"
        verbose: bool = True

    loss: LossParams = LossParams()
    
    @dataclass
    class GraphParams:
        """Parameters for the graph."""
        update_range_main_graph: float=5
        cut_threshold: float=1.0
        edge_dist_thr_main_graph: float=1
        min_samples_for_training: int=6
        random_sample_num: int=100
        
        vis_node_index: int=10
        label_ext_mode: bool=True 
        extraction_store_folder: str='LabelExtraction'
        use_for_training: bool=True
        
    graph: GraphParams=GraphParams()
    
    @dataclass
    class ModelParams:
        name: str = "SimpleMLP"  #  SimpleMLP, SeperateMLP,RndMLP,SeprndMLP
        load_ckpt: Optional[str] = None
        
        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = 384
            hidden_sizes: List[int] = field(default_factory=lambda: [256, 64,256, 2])
            reconstruction: bool = True
            
            def to_dict(self):
                return vars(self)

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()
        
        @dataclass
        class SeperateMLPCfgParams:
            input_size: int = 384
            hidden_sizes: List[int] = field(default_factory=lambda: [256, 64,256, 2])
            
            def to_dict(self):
                return vars(self)
        
        seperate_mlp_cfg: SeperateMLPCfgParams = SeperateMLPCfgParams()
        
        @dataclass
        class RndMLPCfgParams:
            input_size: int = 384
            hidden_sizes_target:List[int] = field(default_factory=lambda: [256,64])
            hidden_sizes_pred:List[int] = field(default_factory=lambda: [256,64])
            pred_head:int=2
            def to_dict(self):
                return vars(self)
        rnd_mlp_cfg: RndMLPCfgParams = RndMLPCfgParams()
        
        @dataclass
        class SeprndMLPCfgParams:
            input_size: int = 384
            hidden_sizes_target:List[int] = field(default_factory=lambda: [256,64])
            hidden_sizes_pred:List[int] = field(default_factory=lambda: [256,64])
            pred_head:int=2
            def to_dict(self):
                return vars(self)
        seprnd_mlp_cfg: SeprndMLPCfgParams = SeprndMLPCfgParams()
    
    model: ModelParams = ModelParams()
    
    @dataclass
    class OfflineParams:
        mode:str='test'
        ckpt_parent_folder:str='results/overlay'
        data_folder:str='results/manager'
        train_data:str='results/manager/train_data.pt'
        nodes_data:str='results/manager/train_nodes.pt'
        image_file:str='image_buffer.pt'
        test_images:bool=True
        test_nodes:bool=True
        
        gt_model:str='SAM' # 'SEEM' or 'SAM'
        SAM_type:str='vit_h'
        SAM_ckpt:str='/media/chen/UDisk1/sam_vit_h_4b8939.pth'
        # SAM_ckpt='/media/chen/UDisk1/sam_hq_vit_h.pth'
    
    offline: OfflineParams = OfflineParams()

def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    return obj

def save_to_yaml(dataclass_instance, filename):
    data_dict = dataclass_to_dict(dataclass_instance)
    with open(filename, 'w') as file:
        yaml.dump(data_dict, file)

if __name__=="__main__":
    params=ParamCollection()
    save_to_yaml(params,'test.yaml')
    pass