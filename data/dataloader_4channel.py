# -*- coding: utf-8 -*-
import os
from pathlib import Path

import imageio  # type: ignore
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.base_dataset import BaseDataset, extract_bayer_channels

to_tensor = transforms.Compose([transforms.ToTensor()])


class LoadData(BaseDataset):
    def __init__(self, dataset_dir: Path, dslr_scale: int, test: bool = False) -> None:
        super().__init__(dataset_dir, dslr_scale, "huawei_raw", test)

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        raw_image = np.asarray(imageio.imread(self.raw_paths[idx]))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))  # type: ignore

        dslr_image = imageio.imread(self.dslr_dir / self.raw_paths[idx].with_suffix(".jpg").name)
        dslr_image = np.asarray(dslr_image)
        dslr_image = np.float32(np.array(Image.fromarray(dslr_image).resize((self.scale, self.scale)))) / 255.0
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        return raw_image, dslr_image, str(idx)  # type: ignore
