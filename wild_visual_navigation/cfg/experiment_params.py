#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from dataclasses import dataclass, field
from typing import List, Optional
from typing import Any
import os
from wild_visual_navigation.cfg import get_global_env_params, GlobalEnvironmentParams


@dataclass
class ExperimentParams:
    env: GlobalEnvironmentParams = get_global_env_params(os.environ.get("ENV_WORKSTATION_NAME", "default"))

    @dataclass
    class GeneralParams:
        name: str = "debug/debug"
        timestamp: bool = True
        tag_list: List[str] = field(default_factory=lambda: ["debug"])
        skip_train: bool = False
        store_model_every_n_steps: Optional[int] = None
        store_model_every_n_steps_key: Optional[str] = None
        log_to_disk: bool = True
        model_path: Optional[str] = None
        log_confidence: bool = True
        use_threshold: bool = True

    general: GeneralParams = GeneralParams()

    @dataclass
    class LoggerParams:
        name: str = "neptune"
        wandb_entity: str = "wild_visual_navigation"
        wandb_project_name: str = "wild_visual_navigation"
        neptune_project_name: str = "ASL/WVN"

    logger: LoggerParams = LoggerParams()

    @dataclass
    class OptimizerParams:
        name: str = "ADAM"
        lr: float = 0.001

    optimizer: OptimizerParams = OptimizerParams()

    @dataclass
    class LossParams:
        anomaly_balanced: bool = True
        w_trav: float = 0.03
        w_reco: float = 0.5
        w_temp: float = 0.0  # 0.75
        method: str = "latest_measurement"
        confidence_std_factor: float = 0.5
        trav_cross_entropy: bool = False

    loss: LossParams = LossParams()

    @dataclass
    class LossAnomalyParams:
        method: str = "latest_measurement"
        confidence_std_factor: float = 0.5

    loss_anomaly: LossAnomalyParams = LossAnomalyParams()

    @dataclass
    class TrainerParams:
        default_root_dir: Optional[str] = None
        precision: int = 32
        accumulate_grad_batches: int = 1
        fast_dev_run: bool = False
        limit_train_batches: float = 1.0
        limit_val_batches: float = 1.0
        limit_test_batches: float = 1.0
        max_epochs: Optional[int] = None
        profiler: Any = False
        num_sanity_val_steps: int = 0
        check_val_every_n_epoch: int = 1
        enable_checkpointing: bool = True
        max_steps: int = 1000
        enable_progress_bar: bool = True
        weights_summary: Optional[str] = "top"
        progress_bar_refresh_rate: Optional[int] = None
        gpus: int = -1

    trainer: TrainerParams = TrainerParams()

    @dataclass
    class AblationDataModuleParams:
        batch_size: int = 8
        num_workers: int = 0
        env: str = "forest"
        feature_key: str = "slic100_dino224_16"
        test_equals_val: bool = False
        val_equals_test: bool = False
        test_all_datasets: bool = False
        training_data_percentage: int = 100
        training_in_memory: bool = True

    ablation_data_module: AblationDataModuleParams = AblationDataModuleParams()

    @dataclass
    class ModelParams:
        name: str = "SimpleMLP"  # LinearRnvp, SimpleMLP, SimpleGCN, DoubleMLP
        load_ckpt: Optional[str] = None

        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = 90  # 90 for stego, 384 for dino
            hidden_sizes: List[int] = field(default_factory=lambda: [256, 32, 1])
            reconstruction: bool = True

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()

        @dataclass
        class DoubleMlpCfgParams:
            input_size: int = 384
            hidden_sizes: List[int] = field(default_factory=lambda: [64, 32, 1])

        double_mlp_cfg: DoubleMlpCfgParams = DoubleMlpCfgParams()

        @dataclass
        class SimpleGcnCfgParams:
            input_size: int = 384
            reconstruction: bool = True
            hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 1])

        simple_gcn_cfg: SimpleGcnCfgParams = SimpleGcnCfgParams()

        @dataclass
        class LinearRnvpCfgParams:
            input_size: int = 384
            coupling_topology: List[int] = field(default_factory=lambda: [200])
            mask_type: str = "odds"
            conditioning_size: int = 0
            use_permutation: bool = True
            single_function: bool = False

        linear_rnvp_cfg: LinearRnvpCfgParams = LinearRnvpCfgParams()

    model: ModelParams = ModelParams()

    @dataclass
    class LrMonitorParams:
        logging_interval: str = "step"

    lr_monitor: LrMonitorParams = LrMonitorParams()

    @dataclass
    class CbEarlyStoppingParams:
        active: bool = False

    cb_early_stopping: CbEarlyStoppingParams = CbEarlyStoppingParams()

    @dataclass
    class CbCheckpointParams:
        active: bool = True

    cb_checkpoint: CbCheckpointParams = CbCheckpointParams()

    @dataclass
    class VisuParams:
        train: int = 0
        val: int = 0
        test: int = 0
        log_test_video: bool = False
        log_val_video: bool = False
        log_train_video: bool = False
        log_every_n_epochs: int = 5

        @dataclass
        class LearningVisuParams:
            p_visu: Optional[str] = None
            store: bool = True
            log: bool = True

        learning_visu: LearningVisuParams = LearningVisuParams()

    visu: VisuParams = VisuParams()
