# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import ensemble_pillow

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

    def __init__(self,
                 dataset_dir: str,
                 dataset_size: int,
                 dslr_scale: int,
                 test: bool = False,
                 if_rotate: bool = True,
                 if_filp: bool = True,
                 is_ensemble: bool = False,
                 is_rescale: bool = False):
        self.is_ensemble = is_ensemble
        self.is_test = test
        self.if_rotate = if_rotate
        self.if_filp = if_filp
        self.is_rescale = is_rescale
        if self.is_test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'test_vis')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'train_vis')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.scale = dslr_scale

        self.tf1 = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
        ])
        self.tf2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
        ])

        self.rescale = transforms.Compose([
            transforms.Resize((self.scale, self.scale)),
        ])

        self.toTensor = transforms.Compose([transforms.ToTensor()])
        self.rotate = transforms.Compose([transforms.RandomRotation(degrees=(-45, 45))])

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        raw_image = Image.open(os.path.join(self.raw_dir, str(idx) + ".png"))
        dslr_image = Image.open(os.path.join(self.dslr_dir, str(idx) + ".jpg"))

        if not self.is_test:
            if self.if_rotate:
                p = random.randint(0, 2)
                if p == 0:
                    raw_image = self.tf1(raw_image)
                    dslr_image = self.tf1(dslr_image)
                elif p == 1:
                    raw_image = self.tf2(raw_image)
                    dslr_image = self.tf2(dslr_image)

        if self.is_rescale:
            raw_image = self.rescale(raw_image)
            dslr_image = self.rescale(dslr_image)

        if self.is_ensemble:
            raw_image = ensemble_pillow(raw_image)  # type: ignore
            raw_image = [self.toTensor(x) for x in raw_image]
        else:
            raw_image = self.toTensor(raw_image)

        dslr_image = self.toTensor(dslr_image)

        return raw_image, dslr_image, str(idx)


class LoadData_real(Dataset):

    def __init__(self, dataset_dir: str, is_ensemble: bool = False) -> None:
        self.is_ensemble = is_ensemble

        self.raw_dir = dataset_dir

        # self.dataset_size = 670
        self.dataset_size = 42

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        raw_image = Image.open(os.path.join(self.raw_dir, str(idx + 1) + ".png"))

        if self.is_ensemble:
            raw_image = ensemble_pillow(raw_image)  # type: ignore
            raw_image = [self.toTensor(x) for x in raw_image]
        else:
            raw_image = self.toTensor(raw_image)

        return raw_image, str(idx + 1)
