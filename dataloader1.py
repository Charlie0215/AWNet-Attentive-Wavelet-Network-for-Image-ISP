# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import PIL.Image as Image
import torch
import os
import random
from prefetch_generator import BackgroundGenerator

to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir, dataset_size, dslr_scale, test=False):

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'test_vis')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'train_vis')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.scale = 224#dslr_scale
        self.test = test

        self.tf1 = transforms.Compose(
            [
                transforms.Resize((self.scale, self.scale)),
                transforms.RandomVerticalFlip(p=1), 
            ]
        )
        self.tf2 = transforms.Compose(
            [
                transforms.Resize((self.scale, self.scale)),
                transforms.RandomHorizontalFlip(p=1), 
            ]
        )
        self.tf3 = transforms.Compose(
            [
                transforms.Resize((self.scale, self.scale)),
            ]
        )
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw_image = Image.open(os.path.join(self.raw_dir, str(idx) + ".png"))
        dslr_image = Image.open(os.path.join(self.dslr_dir, str(idx) + ".jpg"))

        p = random.randint(0, 2)
        if p == 1:
            raw_image = self.tf1(raw_image)
            dslr_image = self.tf1(dslr_image)
        elif p == 2:
            raw_image = self.tf2(raw_image)
            dslr_image = self.tf2(dslr_image)
        else:
            raw_image = self.tf3(raw_image)
            dslr_image = self.tf3(dslr_image)

        raw_image = self.toTensor(raw_image).float()
        dslr_image = self.toTensor(dslr_image)

        return raw_image, dslr_image, str(idx)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class LoadVisualData(Dataset):

    def __init__(self, data_dir, size, scale, level, full_resolution=False):

        self.raw_dir = os.path.join(data_dir, 'test', 'huawei_full_resolution')

        self.dataset_size = size
        self.scale = scale
        self.level = level
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)

        if level > 1 or full_resolution:
            self.image_height = 1440
            self.image_width = 1984
        elif level > 0:
            self.image_height = 1280
            self.image_width = 1280
        else:
            self.image_height = 960
            self.image_width = 960

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = extract_bayer_channels(raw_image)

        if self.level > 1 or self.full_resolution:
            raw_image = raw_image[0:self.image_height, 0:self.image_width, :]
        elif self.level > 0:
            raw_image = raw_image[80:self.image_height + 80, 352:self.image_width + 352, :]
        else:
            raw_image = raw_image[240:self.image_height + 240, 512:self.image_width + 512, :]

        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image
