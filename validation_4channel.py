import torch
import torch.nn as nn
from models.model_4channel import AWNet

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from config import trainConfig
import numpy as np
import imageio
import PIL.Image as Image
import time
import os
from utils import validation, fun_ensemble_back, save_validation_image, fun_ensemble, fun_ensemble_numpy

ENSEMBLE = False

class wrapped_4_channel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = AWNet(4, 3, block=[3, 3, 3, 4, 4])

    def forward(self, x):
        return self.module(x)

class LoadData_real(Dataset):

    def __init__(self, dataset_dir, is_ensemble=False):
        self.is_ensemble = is_ensemble

        self.raw_dir = os.path.join(dataset_dir, 'AIM2020_ISP_fullres_test_raw')

        self.dataset_size = 42

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        idx = idx + 1
        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = extract_bayer_channels(raw_image)

        if self.is_ensemble:

            raw_image = fun_ensemble_numpy(raw_image)
            raw_image = [torch.from_numpy(x.transpose((2, 0, 1)).copy()) for x in raw_image]

        else:
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)).copy())

        return raw_image, str(idx)

def extract_bayer_channels(raw):
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm

def test():
    net1 = wrapped_4_channel()

    net1.load_state_dict(
        torch.load('{}/weight_4channel_best.pkl'.format(trainConfig.save_best), map_location="cpu")["model_state"])
    print('weight loaded.')

    test_dataset = LoadData_real(trainConfig.data_dir, is_ensemble=ENSEMBLE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=False, drop_last=False)

    net1.eval()
    save_folder = './result_fullres_4channel/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for batch_id, val_data in enumerate(test_loader):

        with torch.no_grad():
            raw_image, image_name = val_data
            if isinstance(raw_image, list):
                print('ensemble')
                y1 = [net1(i)[0][0] for i in raw_image]
                y1 = fun_ensemble_back(y1)
                print(y1.shape)
                
            else:
                y1, _ = net1(raw_image)
                y = y1[0]
        if ENSEMBLE:
            save_validation_image(y, image_name, save_folder)
        else:
            save_validation_image(y, image_name, save_folder)


if __name__ == '__main__':
    test()
