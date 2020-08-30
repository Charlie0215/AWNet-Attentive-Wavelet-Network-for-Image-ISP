import torch
import torch.nn as nn
from models.model_4channel import AWNet as gen1
from models.model_3channel import AWNet as gen2
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from config import trainConfig
import numpy as np
import imageio
import PIL.Image as Image
import time
import os
from utils import fun_ensemble_back, save_validation_image, fun_ensemble, to_psnr, to_ssim_skimage, fun_ensemble_numpy

ENSEMBLE = False


class wrapped_3_channel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = gen2(3, 3, block=[3, 3, 3, 4, 4])

    def forward(self, x):
        return self.module(x)


class wrapped_4_channel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = gen1(4, 3, block=[3, 3, 3, 4, 4])

    def forward(self, x):
        return self.module(x)


def extract_bayer_channels(raw):
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData_real(Dataset):
    def __init__(self, dataset_dir, is_ensemble=False):
        self.is_ensemble = is_ensemble


        self.raw_dir1 = os.path.join(dataset_dir,
                                     'AIM2020_ISP_fullres_test_raw')
        self.raw_dir2 = os.path.join(dataset_dir,
                                     'AIM2020_ISP_fullres_test_raw_pseudo_demosaicing')
        self.dataset_size = 42

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        idx = idx + 1
        raw_image1 = np.asarray(
            imageio.imread(os.path.join(self.raw_dir1,
                                        str(idx) + '.png')))
        raw_image1 = extract_bayer_channels(raw_image1)

        raw_image2 = Image.open(os.path.join(self.raw_dir2, str(idx) + ".png"))

        if self.is_ensemble:

            raw_image1 = fun_ensemble_numpy(raw_image1)
            raw_image1 = [
                torch.from_numpy(x.transpose((2, 0, 1)).copy())
                for x in raw_image1
            ]

            raw_image2 = fun_ensemble(raw_image2)
            raw_image2 = [self.toTensor(x) for x in raw_image2]

        else:
            raw_image1 = torch.from_numpy(
                raw_image1.transpose((2, 0, 1)).copy())
            raw_image2 = self.toTensor(raw_image2)

        return raw_image1, raw_image2, str(idx)


def test():
    net1 = wrapped_4_channel()
    net2 = wrapped_3_channel()

    # Reload
    net1.load_state_dict(
        torch.load(
            '{}/weight_4channel_best.pkl'.format(trainConfig.save_best),
            map_location="cpu")["model_state"])
    net2.load_state_dict(
        torch.load(
            '{}/weight_3channel_best.pkl'.format(trainConfig.save_best),
            map_location="cpu")["model_state"])
    print('weight loaded.')

    test_dataset = LoadData_real(trainConfig.data_dir, is_ensemble=ENSEMBLE)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False)

    net1.eval()
    net2.eval()
    save_folder1 = './final_result_fullres/'
    if not os.path.exists(save_folder1):
        os.makedirs(save_folder1)

    for batch_id, val_data in enumerate(test_loader):

        with torch.no_grad():
            raw_image1, raw_image2, image_name = val_data
            if isinstance(raw_image1, list):
                print('ensemble')
                y1 = [net1(i)[0][0] for i in raw_image1]
                y1 = fun_ensemble_back(y1)

                y2 = [net2(i)[0][0] for i in raw_image2]
                y2 = fun_ensemble_back(y2)
                y = (y1 + y2) / 2
            else:
                y1, _ = net1(raw_image1)
                y2, _ = net2(raw_image2)
                y = torch.zeros_like(y1[0])
                y = (y1[0] + y2[0]) / 2
        if ENSEMBLE:
            save_validation_image(y, image_name, save_folder1)
        else:
            save_validation_image(y, image_name, save_folder1)


if __name__ == '__main__':
    test()
