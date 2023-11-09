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
        phy_decoder_output_topic:str='/vd_pipeline/phy_decoder_output'

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
        rear_camera_in_base= np.array([[-0.98962361  ,0.          ,0.14368394 ,-0.3537],
                                        [ 0.          ,0.99999982 , 0.         , 0.    ],
                                        [-0.14368394  ,0.         ,-0.98962343 , 0.1634],
                                        [ 0.          ,0.         ,0.          ,1.      ]])
        pass
    roscfg: RosParams=RosParams()
    
    @dataclass
    class ThreadParams:
        """Parameters for the threads."""
        image_callback_rate: float=2.0
        proprio_callback_rate: float=10.0
        learning_rate: float=1.0
        logging_rate: float=2.0
    
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
        max_dist_sub_graph: float=3
        edge_dist_thr_sub_graph: float=0.2
        edge_dist_thr_main_graph: float=1
        min_samples_for_training: int=10
        vis_node_index: int=10
        label_ext_mode: bool=False
        extraction_store_folder: str='LabelExtraction'
        use_for_training: bool=True
        
    graph: GraphParams=GraphParams()