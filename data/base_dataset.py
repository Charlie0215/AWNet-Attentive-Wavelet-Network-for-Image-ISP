# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def extract_bayer_channels(raw: np.ndarray) -> np.ndarray:
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class BaseDataset(Dataset):
    def __init__(self, dataset_dir: Path, dslr_scale: int, gt_image_folder_name: str, test: bool = False) -> None:
        self.is_test = test
        self.scale = dslr_scale

        if self.is_test:
            self.raw_dir = dataset_dir / "test" / gt_image_folder_name
            self.dslr_dir = dataset_dir / "test" / "canon"
        else:
            self.raw_dir = dataset_dir / "train" / gt_image_folder_name
            self.dslr_dir = dataset_dir / "train" / "canon"
        self.raw_paths = sorted(self.raw_dir.glob("*.png"))
