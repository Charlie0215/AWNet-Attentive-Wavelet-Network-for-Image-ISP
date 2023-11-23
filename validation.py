import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from params import PipelineParams
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse

from models.model_3channel import AWNetThreeChannel
from utils import disassemble_ensembled_img, ensemble_pillow, save_ensemble_image
from pathlib import Path
from utils import load_yaml_config

ENSEMBLE = False


class wrapped_3_channel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.module = AWNetThreeChannel(3, num_gcrdb=[3, 3, 3, 4, 4])

    def forward(self, x: torch.Tensor) -> None:
        return self.module(x)


class LoadData_real(Dataset):

    def __init__(self, dataset_dir: str, is_ensemble: bool = False) -> None:
        self.is_ensemble = is_ensemble
        self.raw_dir = os.path.join(dataset_dir, 'AIM2020_ISP_fullres_test_raw_pseudo_demosaicing')
        self.dataset_size = 42
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        idx = idx + 1
        raw_image = Image.open(os.path.join(self.raw_dir, str(idx) + ".png"))

        if self.is_ensemble:
            raw_image = ensemble_pillow(raw_image)
            raw_image = [self.toTensor(x) for x in raw_image]

        else:
            raw_image = self.toTensor(raw_image)

        return raw_image, str(idx)


def extract_bayer_channels(raw: np.ndarray) -> np.ndarray:
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


def test(args: argparse.ArgumentParser) -> None:
    params: PipelineParams = load_yaml_config(PipelineParams, args.config_file)
    log_folder: Path = args.log_folder
    saving_folder: Path = args.saving_folder
    net = AWNetThreeChannel(params.awnet_model_params.input_num_channels, params.awnet_model_params.num_gcrdb)
    
    weight_path = log_folder / "checkpoints" / params.training_params.best_model_name
    net.load_state_dict(torch.load(weight_path, map_location="cpu")["model_state"])  # type: ignore
    print(f"Loaded weight from: {weight_path}")

    test_dataset = LoadData_real(params.dataset_params.train_dataset_dir, is_ensemble=ENSEMBLE)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=params.dataloader_params.test.batch_size,
                             shuffle=params.dataloader_params.test.shuffle,
                             num_workers=params.dataloader_params.test.num_workers,
                             pin_memory=params.dataloader_params.test.pin_memory,
                             drop_last=params.dataloader_params.test.drop_last)

    net.eval()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for _, val_data in enumerate(test_loader):

        with torch.no_grad():  # type: ignore
            x, image_name = val_data
            if isinstance(x, list):
                y = [net(i)[0][0] for i in x]
                y = disassemble_ensembled_img(y)  # type: ignore
            else:
                y, _ = net(x)
        if ENSEMBLE:
            save_ensemble_image(y, image_name, saving_folder)  # type: ignore
        else:
            save_ensemble_image(y[0], image_name, saving_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for the training pipeline.")
    parser.add_argument("-c", "--config-file", type=Path, help="Path to the pipeline config.")
    parser.add_argument("-l", "--log-folder", type=Path, help="Path to the log folder that saves the training results.")
    parser.add_argument("-s", "--saving-folder", type=Path, help="Path to the folder that saves predicted images.")
    args = parser.parse_args()
    test(args)
