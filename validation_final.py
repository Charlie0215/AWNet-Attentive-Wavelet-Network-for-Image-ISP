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

from models.model_3channel import AWNetThreeChannel as gen2
from models.model_4channel import AWNetFourChannel as gen1
from utils import disassemble_ensembled_img, ensemble_ndarray, ensemble_pillow, save_ensemble_image

ENSEMBLE = True


def extract_bayer_channels(raw: np.ndarray) -> np.ndarray:
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData_real(Dataset):

    def __init__(self, dataset_dir: str, is_ensemble: bool = False) -> None:
        self.is_ensemble = is_ensemble
        self.raw_dir1 = os.path.join(dataset_dir, 'AIM2020_ISP_test_raw')
        self.raw_dir2 = os.path.join(dataset_dir, 'AIM2020_ISP_test_pseudo_demosaicing')
        self.dataset_size = 1342

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tehnsor, str]:
        raw_image1 = np.asarray(imageio.imread(os.path.join(self.raw_dir1, str(idx) + '.png')))
        raw_image1 = extract_bayer_channels(raw_image1)

        raw_image2 = Image.open(os.path.join(self.raw_dir2, str(idx) + ".png"))

        if self.is_ensemble:

            raw_image1 = ensemble_ndarray(raw_image1)  # type: ignore
            raw_image1 = [
                torch.from_numpy(x.transpose(  # type: ignore
                    (2, 0, 1)).copy()) for x in raw_image1
            ]

            raw_image2 = ensemble_pillow(raw_image2)
            raw_image2 = [self.toTensor(x) for x in raw_image2]

        else:
            raw_image1 = torch.from_numpy(  # type: ignore
                raw_image1.transpose((2, 0, 1)).copy())
            raw_image2 = self.toTensor(raw_image2)

        return raw_image1, raw_image2, str(idx)  # type: ignore


def test() -> None:
    device_ids = [0]
    print('using device: {}'.format(device_ids))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net1 = gen1(4, 3, block=[3, 3, 3, 4, 4]).to(device)
    net2 = gen2(3, 3, num_gcrdb=[3, 3, 3, 4, 4]).to(device)
    net1 = nn.DataParallel(net1, device_ids=device_ids)  # type: ignore
    net2 = nn.DataParallel(net2, device_ids=device_ids)  # type: ignore

    # Reload
    net1.load_state_dict(
        torch.load(f"{trainConfig.save_best}/weight_4channel_best.pkl",
                   map_location="cuda:0")["model_state"])  # type: ignore
    net2.load_state_dict(
        torch.load("{trainConfig.save_best}/weight_3channel_best.pkl",
                   map_location="cuda:0")["model_state"])  # type: ignore
    print('weight loaded.')

    test_dataset = LoadData_real(trainConfig.data_dir, is_ensemble=ENSEMBLE)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False,
                             num_workers=12,
                             pin_memory=True,
                             drop_last=False)

    save_folder = './final_result/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for batch_id, val_data in enumerate(test_loader):

        with torch.no_grad():  # type: ignore
            raw_image1, raw_image2, image_name = val_data
            if isinstance(raw_image1, list):
                print('ensemble')
                raw_image1 = [i.to(device) for i in raw_image1]
                y1 = [net1(i)[0][0] for i in raw_image1]
                y1 = disassemble_ensembled_img(y1)  # type: ignore

                raw_image2 = [i.to(device) for i in raw_image2]
                y2 = [net2(i)[0][0] for i in raw_image2]
                y2 = disassemble_ensembled_img(y2)  # type: ignore
                y = (y1 + y2) / 2  # type: ignore
            else:
                raw_image1 = raw_image1.to(device, non_blocking=True)
                raw_image2 = raw_image2.to(device, non_blocking=True)
                y1, _ = net1(raw_image1)
                y2, _ = net2(raw_image2)
                y = torch.zeros_like(y1[0])  # type: ignore
                y = (y1[0] + y2[0]) / 2
        if ENSEMBLE:
            save_ensemble_image(y, image_name, save_folder)  # type: ignore
        else:
            save_ensemble_image(y, image_name, save_folder)  # type: ignore


if __name__ == '__main__':
    test()
