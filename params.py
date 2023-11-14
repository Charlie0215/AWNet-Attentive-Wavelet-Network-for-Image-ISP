# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class TrainingParams:
    is_demosaic: bool
    experiment_name: str
    root_log_dir: str
    num_gpus: int
    device_list: list[int]
    learning_rate_milestones: list[float]
    print_loss: bool
    num_epoch: int
    use_pretrained_weight: bool
    best_model_name: str
    logging_level: str

    def __post_init__(self) -> None:
        for i in range(1, len(self.learning_rate_milestones)):
            if self.learning_rate_milestones[i] >= self.learning_rate_milestones[i - 1]:
                raise ValueError("Learning rate milestones should be strictly decreasing, "
                                 f"but we have {self.learning_rate_milestones}")


@dataclass
class DataloaderParams:
    num_workers: int
    pin_memory: bool
    batch_size: int
    shuffle: bool
    drop_last: bool


@dataclass
class PipelineDataloaderParams:
    train: DataloaderParams
    val: DataloaderParams


@dataclass
class AwnetParams:
    input_num_channels: int
    # num of GCRDB blocks for each feature size
    num_gcrdb: list[int]


@dataclass
class DatasetParams:
    resized_size: int
    train_dataset_dir: str
    val_dataset_dir: str


@dataclass
class PipelineParams:
    training_params: TrainingParams
    dataloader_params: PipelineDataloaderParams
    awnet_model_params: AwnetParams
    dataset_params: DatasetParams
