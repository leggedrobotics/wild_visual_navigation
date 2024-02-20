#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
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
    supervision_graph_dist_thr: float  # meters
    confidence_std_factor: float
    min_samples_for_training: int
    network_input_image_height: int
    network_input_image_width: int
    vis_node_index: int

    # Supervision Generator
    untraversable_thr: float

    mission_name: str
    mission_timestamp: bool

    # Threads
    image_callback_rate: float  # hertz
    supervision_callback_rate: float  # hertz
    learning_thread_rate: float  # hertz
    logging_thread_rate: float  # hertz
    load_save_checkpoint_rate: float  # hert

    # Runtime options
    device: str
    mode: Any  # check out comments in the class WVNMode
    colormap: str

    print_image_callback_time: bool
    print_supervision_callback_time: bool
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
    dino_backbone: str  # vit_small, vit_base
    slic_num_components: int

    # ConfidenceGenerator
    confidence_std_factor: float

    # Output setting
    prediction_per_pixel: bool

    # Runtime options
    mode: Any  # check out comments in the class WVNMode
    status_thread_rate: float  # hertz
    device: str
    log_confidence: bool
    verbose: bool

    # Threads
    image_callback_rate: float  # hertz
    load_save_checkpoint_rate: float  # hertz
