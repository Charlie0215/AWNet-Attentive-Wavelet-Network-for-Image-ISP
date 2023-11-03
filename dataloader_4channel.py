# -*- coding: utf-8 -*-
import os

import imageio  # type: ignore
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

to_tensor = transforms.Compose([transforms.ToTensor()])


def extract_bayer_channels(raw: np.ndarray) -> np.ndarray:
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir: str, dataset_size: int, dslr_scale: int, test: bool = False) -> None:

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.scale = dslr_scale
        self.test = test

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))  # type: ignore

        dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg"))
        dslr_image = np.asarray(dslr_image)
        dslr_img_shape = dslr_image.shape
        dslr_image = np.float32(
            np.array(
                Image.fromarray(dslr_image).resize(
                    (dslr_img_shape[0] // self.scale, dslr_img_shape[1] // self.scale)))) / 255.0
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        return raw_image, dslr_image, str(idx)  # type: ignore
