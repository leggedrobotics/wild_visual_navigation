""" 
This file contains the all configurations for the Wild-Visual-Navigation project.
 """
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from simple_parsing.helpers import Serializable
import numpy as np
@dataclass
class ParamCollection(Serializable):
    """A collection of parameters."""
    @dataclass
    class GeneralParams:
        """General parameters for the experiment."""
        name: str='debug/debug'
        timestamp: bool=True
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
        learning_rate: float=1.0
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
    
    feat: FeatParams=FeatParams()

    @dataclass
    class LossParams:
        anomaly_balanced: bool = True
        w_trav: float = 0.03
        w_reco: float = 0.5
        w_temp: float = 0.0  # 0.75
        method: str = "running_mean"
        confidence_std_factor: float = 2
        log_enabled: bool = False
        log_folder: str = "/tmp"
        verbose: bool = True
        trav_cross_entropy: bool = False

    loss: LossParams = LossParams()
    
    @dataclass
    class GraphParams:
        """Parameters for the graph."""
        update_range_main_graph: float=5
        cut_threshold: float=2.0
        edge_dist_thr_main_graph: float=1
        min_samples_for_training: int=10
        vis_node_index: int=10
        label_ext_mode: bool=False
        extraction_store_folder: str='LabelExtraction'
        use_for_training: bool=True
        
    graph: GraphParams=GraphParams()
    
    @dataclass
    class ModelParams:
        name: str = "SimpleMLP"  #  SimpleMLP
        load_ckpt: Optional[str] = None
        
        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = 384
            hidden_sizes: List[int] = field(default_factory=lambda: [256, 32, 1])
            reconstruction: bool = True
            
            def to_dict(self):
                return vars(self)

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()
    
    model: ModelParams = ModelParams()