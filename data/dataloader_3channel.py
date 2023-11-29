# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.base_dataset import BaseDataset
from utils import ensemble_ndarray

to_tensor = transforms.Compose([transforms.ToTensor()])


class LoadData(BaseDataset):
    def __init__(
        self,
        dataset_dir: Path,
        dslr_scale: int,
        test: bool = False,
        is_rotate: bool = True,
        is_filp: bool = True,
        is_ensemble: bool = False,
        is_rescale: bool = False,
    ):
        super().__init__(dataset_dir, dslr_scale, "demosaiced", test)

        self.is_ensemble = is_ensemble
        self.is_rotate = is_rotate
        self.is_filp = is_filp
        self.is_rescale = is_rescale

        self.tf1 = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=1),
            ]
        )
        self.tf2 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1),
            ]
        )

        self.rescale = transforms.Compose(
            [
                transforms.Resize((self.scale, self.scale)),
            ]
        )

        self.toTensor = transforms.Compose([transforms.ToTensor()])
        self.rotate = transforms.Compose([transforms.RandomRotation(degrees=(-45, 45))])

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        raw_image = Image.open(self.raw_paths[idx])
        dslr_image = Image.open(self.dslr_dir / self.raw_paths[idx].with_suffix(".jpg").name)

        if not self.is_test and self.is_rotate:
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
        self.raw_paths = sorted(Path(dataset_dir).glob("*.png"))
        self.is_ensemble = is_ensemble
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        raw_image = Image.open(str(self.raw_paths[idx]))

        if self.is_ensemble:
            raw_image = ensemble_pillow(raw_image)  # type: ignore
            raw_image = [self.toTensor(x) for x in raw_image]
        else:
            raw_image = self.toTensor(raw_image)

        return raw_image, str(idx + 1)
