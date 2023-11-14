# -*- coding: utf-8 -*-
import os
import time

import imageio  # type: ignore
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from config import trainConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.model_4channel import AWNetFourChannel
from utils import disassemble_ensembled_img, ensemble_ndarray, ensemble_pillow, save_ensemble_image

ENSEMBLE = False


class wrapped_4_channel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.module = AWNetFourChannel(4, 3, block=[3, 3, 3, 4, 4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class LoadData_real(Dataset):

    def __init__(self, dataset_dir: str, is_ensemble: bool = False) -> None:
        self.is_ensemble = is_ensemble

        self.raw_dir = os.path.join(dataset_dir, 'AIM2020_ISP_fullres_test_raw')

        self.dataset_size = 42

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        idx = idx + 1
        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = extract_bayer_channels(raw_image)

        if self.is_ensemble:

            raw_image = ensemble_ndarray(raw_image)  # type: ignore
            raw_image = [
                torch.from_numpy(x.transpose(  # type: ignore
                    (2, 0, 1)).copy()) for x in raw_image
            ]

        else:
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)).copy())  # type: ignore

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


def test() -> None:
    net1 = wrapped_4_channel()

    net1.load_state_dict(
        torch.load("{trainConfig.save_best}/weight_4channel_best.pkl",
                   map_location="cpu")["model_state"])  # type: ignore
    print('weight loaded.')

    test_dataset = LoadData_real(trainConfig.data_dir, is_ensemble=ENSEMBLE)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=False,
                             drop_last=False)

    net1.eval()
    save_folder = './result_fullres_4channel/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for _, val_data in enumerate(test_loader):

        with torch.no_grad():  # type: ignore
            raw_image, image_name = val_data
            if isinstance(raw_image, list):
                y1 = [net1(i)[0][0] for i in raw_image]
                y1 = disassemble_ensembled_img(y1)  # type: ignore

            else:
                y1, _ = net1(raw_image)
                y = y1[0]
        if ENSEMBLE:
            save_ensemble_image(y, image_name, save_folder)
        else:
            save_ensemble_image(y, image_name, save_folder)


if __name__ == '__main__':
    test()
