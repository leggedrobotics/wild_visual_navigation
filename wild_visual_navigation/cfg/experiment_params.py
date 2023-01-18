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
        skip_train: bool = False
        store_model_every_n_steps: Optional[int] = None
        store_model_every_n_steps_key: Optional[str] = None
        log_to_disk: bool = True
        model_path: Optional[str] = None

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
        lr: float = 0.001
        adamw_cfg: AdamwCfgParams = AdamwCfgParams()

    optimizer: OptimizerParams = OptimizerParams()

    @dataclass
    class LossParams:
        anomaly_balanced: bool = True
        w_trav: float = 0.4
        w_reco: float = 1.1
        w_temp: float = 0.4
        use_kalman_filter: bool = True
        false_negative_weight: float = 1.0
        confidence_std_factor: float = 2.0

    loss: LossParams = LossParams()

    @dataclass
    class TrainerParams:
        precision: int = 32
        accumulate_grad_batches: int = 1
        fast_dev_run: bool = False
        limit_train_batches: float = 1.0
        limit_val_batches: float = 1.0
        limit_test_batches: float = 1.0
        max_epochs: Optional[int] = 10
        profiler: bool = False
        num_sanity_val_steps: int = 0
        check_val_every_n_epoch: int = 10
        enable_checkpointing: bool = True
        max_steps: int = -1

    trainer: TrainerParams = TrainerParams()

    @dataclass
    class AblationDataModuleParams:
        batch_size: int = 8
        num_workers: int = 0
        env: str = "forest"
        feature_key: str = "slic100_dino112_8"
        test_equals_val: bool = False
        val_equals_test: bool = False
        test_all_datasets: bool = False
        training_data_percentage: int = 100

    ablation_data_module: AblationDataModuleParams = AblationDataModuleParams()

    @dataclass
    class ModelParams:
        name: str = "SimpleGCN"
        load_ckpt: Optional[str] = None

        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = 90
            hidden_sizes: List[int] = field(default_factory=lambda: [64, 32, 1])
            reconstruction: bool = True

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()

        @dataclass
        class SimpleGcnCfgParams:
            input_size: int = 90
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
        log_val_video: bool = False
        log_train_video: bool = False
        log_every_n_epochs: int = 10

        @dataclass
        class LearningVisuParams:
            p_visu: Optional[bool] = None
            store: bool = True
            log: bool = True

        learning_visu: LearningVisuParams = LearningVisuParams()

    visu: VisuParams = VisuParams()

    def verify_params(self):
        if not self.general.log_to_disk:
            assert self.trainer.profiler != "advanced", "Should not be advanced if not logging to disk"
            assert self.cb_checkpoint.active == False, "Should be False if not logging to disk"
