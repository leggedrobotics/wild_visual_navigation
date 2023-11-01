from dataclasses import dataclass
from typing import Dict
from typing import Any


@dataclass
class RosLearningNodeParams:
    # Input topics
    camera_topics: Dict[str, Any]
    robot_state_topic: str
    desired_twist_topic: str

    # Relevant frames
    fixed_frame: str
    base_frame: str
    footprint_frame: str

    # Robot size
    robot_length: float
    robot_width: float
    robot_height: float

    # Traversability estimation params
    traversability_radius: float  # meters
    image_graph_dist_thr: float  # meters
    proprio_graph_dist_thr: float  # meters
    network_input_image_height: int  # 448
    network_input_image_width: int  # 448
    segmentation_type: str
    feature_type: str
    dino_patch_size: int  # 8 or 16; 8 is finer
    slic_num_components: int
    dino_dim: int  # 90 or 384; 384 is better
    confidence_std_factor: float
    scale_traversability: bool  # This parameter needs to be false when using the anomaly detection model
    scale_traversability_max_fpr: float
    min_samples_for_training: int

    traversability_threshold: float
    vis_node_index: int

    # Supervision Generator
    untraversable_thr: float

    mission_name: str
    mission_timestamp: bool

    # Threads
    image_callback_rate: float  # hertz
    proprio_callback_rate: float  # hertz
    learning_thread_rate: float  # hertz
    logging_thread_rate: float  # hertz

    # Runtime options
    device: str
    mode: Any  # check out comments in the class WVNMode
    colormap: str

    print_image_callback_time: bool
    print_proprio_callback_time: bool
    log_time: bool
    log_confidence: bool
    verbose: bool

    extraction_store_folder: str


@dataclass
class RosFeatureExtractorNodeParams:
    # Input topics
    camera_topics: Dict[str, Any]

    # FeatureExtractor
    network_input_image_height: int  # 448
    network_input_image_width: int  # 448
    segmentation_type: str
    feature_type: str
    dino_patch_size: int  # 8 or 16; 8 is finer
    slic_num_components: int
    dino_dim: int  # 90 or 384; 384 is better

    # ConfidenceGenerator
    confidence_std_factor: float
    scale_traversability: bool  # This parameter needs to be false when using the anomaly detection model

    # Output setting
    prediction_per_pixel: bool
    traversability_threshold: float
    clip_to_binary: bool

    # Runtime options
    mode: Any  # check out comments in the class WVNMode
    status_thread_rate: float  # hertz
    device: str
    log_confidence: bool
    verbose: bool
