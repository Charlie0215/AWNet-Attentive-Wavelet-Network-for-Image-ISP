import time
from datetime import datetime
from skimage import measure
import os
import cv2
import numpy as np
from PIL import Image
from math import log10
import torch
import torch.nn.functional as F
import torchvision.utils as utils
import torch.nn as nn
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torchvision.utils as vutils



def display_transform():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])


def get_colors():
    '''
    Dictionary of color map
    '''
    return np.asarray([0, 128, 255])

class time_calculator():

    def __init__(self):
        self.accumulate = 0

    def calculator(self, start_time, end_time, length=0):
        if length == 0:
            total_time = int(end_time - start_time) + self.accumulate
            self.accumulate = total_time
        else:
            total_time = int(end_time - start_time) * length
        total_time_min = total_time // 60
        total_time_sec = total_time % 60
        total_time_hr = total_time // 2400
        return total_time_hr, total_time_min, total_time_sec

def writer_add_image(dir, writer, image, iter):
    '''
    tensorboard image writer
    '''
    x = vutils.make_grid(image, nrow=4)
    writer.add_image(dir, x, iter)


def save_image(target, preds, img_name, root):
    '''
    : img: image to be saved
    : img_name: image name
    '''
    target = torch.split(target, 1, dim=0)
    preds = torch.split(preds, 1, dim=0)
    batch_num = len(preds)

    for ind in range(batch_num):
        vutils.save_image(target[ind], root + '{}'.format(img_name[ind].split('.png')[0] + '_pred.png'))
        vutils.save_image(preds[ind], root + '{}'.format(img_name[ind].split('.png')[0] + '_alpha.png'))


def validation(net, val_data_loader, device, save_tag=False):
    psnr_list = []
    ssim_list = []
    mse_list = []
    sad_list = []
    save_folder = os.path.join(
        './results', 'result_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            x, target, image_name = val_data
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            y = net(x)

        psnr_list.extend(to_psnr(y, target))
        ssim_list.extend(to_ssim_skimage(y, target))
        # Save image
        if save_tag:
            save_image(target, y, image_name, save_folder)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    return avr_psnr, avr_ssim


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True,
                                      sigma=1.5, gaussian_weights=True, use_sample_covariance=False) for ind in range(len(dehaze_list))]

    return ssim_list

def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))
    # write training log
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_Image_PSNR: {7:.2f}, Val_Image_SSIM: {8:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)

def adjust_learning_rate(optimizer, scheduler, epoch, learning_rate, writer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch > 0: #and epoch < 16:
        if epoch % 6 == 0:
            learning_rate = scheduler.get_lr()[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                print('Learning rate sets to {}.'.format(param_group['lr']))
                scheduler.step()
            writer.add_scalars('lr/train_lr_group',
               {'lr': learning_rate,
               }, epoch)
    return learning_rate

def poly_learning_decay(optimizer, iter, total_epoch, loader_length, writer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    max_iteration = total_epoch * loader_length
    learning_rate = optimizer.param_groups[0]['lr']
    learning_rate = learning_rate * (1 - iter/max_iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    writer.add_scalars('lr/train_lr_group',
        {'lr': learning_rate,
        }, iter)
    return learning_rate