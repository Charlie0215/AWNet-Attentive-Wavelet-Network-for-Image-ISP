# -*- coding: utf-8 -*-
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from config import trainConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.model_3channel import AWNetThreeChannel
from utils import disassemble_ensembled_img, ensemble_pillow, save_ensemble_image

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

        self.raw_dir = os.path.join(dataset_dir, "AIM2020_ISP_fullres_test_raw_pseudo_demosaicing")

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


def test() -> None:
    device_ids = [0]
    print(f"Using device: {device_ids}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = wrapped_3_channel()
    # Reload

    net.load_state_dict(
        torch.load(f"{trainConfig.save_best}/weight_3channel_best.pkl", map_location="cpu")["model_state"]
    )  # type: ignore
    print("weight loaded.")

    test_dataset = LoadData_real(trainConfig.data_dir, is_ensemble=ENSEMBLE)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False
    )

    net.eval()
    save_folder = "./result_fullres_3channel/"
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
            save_ensemble_image(y, image_name, save_folder)  # type: ignore
        else:
            save_ensemble_image(y[0], image_name, save_folder)


if __name__ == "__main__":
    test()
