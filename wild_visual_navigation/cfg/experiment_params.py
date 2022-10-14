from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from simple_parsing.helpers import Serializable


@dataclass
class ExperimentParams(Serializable):
    @dataclass
    class GeneralParams:
        name: str = "simple_gcn/debug"
        timestamp: bool = False
        tag_list: List[str] = field(default_factory=lambda: ["debug"])

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
        @dataclass
        class AdamwCfgParams:
            momentum: float = 0.9
            weight_decay: float = 4.0e-05

        name: str = "ADAMW"
        lr: float = 0.005
        adamw_cfg: AdamwCfgParams = AdamwCfgParams()

    optimizer: OptimizerParams = OptimizerParams()

    @dataclass
    class LossParams:
        anomaly_blanced: bool = True
        w_trav: float = 0.25
        w_reco: float = 1.0
        w_temp: float = 0.0

    loss: LossParams = LossParams()

    @dataclass
    class TrainerParams:
        precision: int = 32
        accumulate_grad_batches: int = 1
        fast_dev_run: bool = False
        limit_train_batches: float = 1.0
        limit_val_batches: float = 1.0
        limit_test_batches: float = 1.0
        max_epochs: int = 10
        profiler: bool = False
        num_sanity_val_steps: int = 0
        check_val_every_n_epoch: int = 1

    trainer: TrainerParams = TrainerParams()

    @dataclass
    class DataModuleParams:
        visu: bool = True
        batch_size: int = 8
        num_workers: int = 0
        dataset_folder: str = "results/perugia_forest_uphill_loop"

    data_module: DataModuleParams = DataModuleParams()

    @dataclass
    class AbblationDataModuleParams:
        batch_size: int = 8
        num_workers: int = 0
        env: str = "forest"
        feature_key: str = "slic_dino"
        test_equals_val: bool = False

    abblation_data_module: AbblationDataModuleParams = AbblationDataModuleParams()

    @dataclass
    class ModelParams:
        name: str = "SimpleGCN"

        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = 90
            hidden_sizes: List[int] = field(default_factory=lambda: [64, 32, 1])
            reconstruction: bool = True

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()

        @dataclass
        class SimpleGcnCfgParams:
            num_node_features: int = 90
            reconstruction: bool = True
            hidden_sizes: List[int] = field(default_factory=lambda: [64, 32, 1])

        simple_gcn_cfg: SimpleGcnCfgParams = SimpleGcnCfgParams()

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
        train: int = 2
        val: int = 2
        test: int = 2
        log_test_video: bool = False
        log_val_video: bool = True
        log_train_video: bool = False
        log_every_n_epochs: int = 10

    visu: VisuParams = VisuParams()
