# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from math import log10
from typing import Optional, Type

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from PIL import Image
from skimage import measure
from torchvision.transforms import Compose, ToPILImage, ToTensor


def display_transform() -> torchvision.transforms.Compose:
    return Compose([ToPILImage(), ToTensor()])


def get_colors() -> np.ndarray:
    '''
    Dictionary of color map
    '''
    return np.asarray([0, 128, 255])


def writer_add_image(dir: str, writer: torch.utils.tensorboard.SummaryWriter, image: np.ndarray, iter: int) -> None:
    '''
    tensorboard image writer
    '''
    x = vutils.make_grid(image, nrow=4)
    writer.add_image(dir, x, iter)


def save_image(target: torch.Tensor, preds: torch.Tensor, img_name: str, root: str) -> None:
    '''
    : img: image to be saved
    : img_name: image name
    '''
    target = torch.split(target, 1, dim=0)  # type: ignore
    preds = torch.split(preds, 1, dim=0)  # type: ignore
    batch_num = len(preds)

    for ind in range(batch_num):
        vutils.save_image(target[ind], root + f"{img_name[ind].split('.png')[0] + '_target.png'}")
        vutils.save_image(preds[ind], root + f"{img_name[ind].split('.png')[0] + '_pred.png'}")


def save_validation_image(preds: torch.Tensor, img_name: str, save_folder: str) -> None:
    '''
    : img: image to be saved
    : img_name: image name
    '''
    preds = torch.split(preds, 1, dim=0)  # type: ignore
    batch_num = len(preds)

    for ind in range(batch_num):
        print('saving {}'.format(img_name[ind]))
        vutils.save_image(preds[ind], save_folder + '{}'.format(img_name[ind].split('.png')[0] + '.png'))


def ensemble_pillow(img: PIL.PngImagePlugin.PngImageFile) -> list[PIL.PngImagePlugin.PngImageFile]:
    imgs = [
        img,  # 0
        img.rotate(90),  # 1
        img.rotate(180),  # 2
        img.rotate(270),  # 3
        img.transpose(Image.FLIP_TOP_BOTTOM),  # 4
        img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM),  # 5
        img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM),  # 6
        img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)
    ]  # 7
    return imgs


def ensemble_ndarray(img: np.ndarray) -> list[np.ndarray]:
    imgs = [
        img,  # 0
        np.rot90(img, k=1, axes=(0, 1)),
        np.rot90(img, k=2, axes=(0, 1)),
        np.rot90(img, k=3, axes=(0, 1)),
        np.flipud(img),  # 0
        np.flipud(np.rot90(img, k=1, axes=(0, 1))),
        np.flipud(np.rot90(img, k=2, axes=(0, 1))),
        np.flipud(np.rot90(img, k=3, axes=(0, 1))),
    ]
    return imgs


def disassemble_ensembled_img(imgs: torch.Tensor) -> torch.Tensor:
    disassembled_img: list[torch.Tensor] = [
        imgs[0], imgs[1].transpose(2, 3).flip(3), imgs[2].flip(2).flip(3), imgs[3].transpose(2, 3).flip(2),
        imgs[4].flip(2), imgs[5].transpose(2, 3), imgs[6].flip(3), imgs[7].transpose(2, 3).flip(2).flip(3)
    ]

    mean_disassembled_img = sum(disassembled_img) / len(imgs)

    return img  # type: ignore


def validation(net: torch.nn.Module,
               val_data_loader: torch.utils.DataLoader,
               device: torch.device,
               texture_net: Optional[torch.nn.Module] = None,
               save_tag: bool = False,
               mode: str = "student",
               is_validation: bool = False,
               is_ensemble: bool = False) -> tuple[float, float]:
    psnr_list = []
    ssim_list = []
    save_folder = os.path.join('./results', 'result_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '/')
    net.eval()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():  # type: ignore
            x, target, image_name = val_data
            target = target.to(device, non_blocking=True)
            if mode == 'orig_size':
                if isinstance(x, list):
                    x = [i.to(device) for i in x]
                    pred_tensors = [net(i)[1] for i in x]
                    y = disassemble_ensembled_img(pred_tensors)  # type: ignore
                else:
                    x = x.to(device, non_blocking=True)
                    _, y, _ = net(x)
                psnr_list.extend(to_psnr(y, target))
                ssim_list.extend(to_ssim_skimage(y, target))  # type: ignore
            elif mode == 'student' or mode == 'teacher':
                if texture_net:
                    x = x.to(device, non_blocking=True)
                    x = texture_net(x)
                    y, _ = net(x)
                    psnr_list.extend(to_psnr(y[0], target))
                    ssim_list.extend(to_ssim_skimage(y[0], target))
                else:
                    if isinstance(x, list):
                        x = [i.to(device) for i in x]
                        y = [net(i)[0][0] for i in x]  # type: ignore
                        y = disassemble_ensembled_img(y)
                        psnr_list.extend(to_psnr(y, target))
                        ssim_list.extend(to_ssim_skimage(y, target))  # type: ignore
                    else:
                        x = x.to(device, non_blocking=True)
                        y, _ = net(x)
                        psnr_list.extend(to_psnr(y[0], target))
                        ssim_list.extend(to_ssim_skimage(y[0], target))

            elif mode == 'texture':
                x = x.to(device, non_blocking=True)
                y = net(x)
                psnr_list.extend(to_psnr(y, target))
                ssim_list.extend(to_ssim_skimage(y, target))  # type: ignore

        # Save image
        if save_tag:
            if mode == 'orig_size':
                save_image(target, y, image_name, save_folder)  # type: ignore
            elif (mode == 'student' or mode == 'teacher') and is_validation == False:
                save_image(target, y[0], image_name, save_folder)
            elif (mode == 'student' or mode == 'teacher') and is_validation == True:
                if is_ensemble:
                    save_validation_image(y, image_name, './validation')  # type: ignore
                else:
                    save_validation_image(y[0], image_name, './validation')
            elif mode == 'texture':
                save_image(target, y, image_name, save_folder)  # type: ignore

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    return avr_psnr, avr_ssim


def to_psnr(dehaze: torch.Tenspr, gt: torch.Tensor) -> list[float]:
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)  # type: ignore
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze: torch.Tensor, gt: torch.Tensor) -> list[float]:
    dehaze_list = torch.split(dehaze, 1, dim=0)  # type: ignore
    gt_list = torch.split(gt, 1, dim=0)  # type: ignore

    dehaze_list_np = [
        dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))
    ]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [
        measure.compare_ssim(
            dehaze_list_np[ind],  # type: ignore
            gt_list_np[ind],
            data_range=1,
            multichannel=True,
            sigma=1.5,
            gaussian_weights=True,
            use_sample_covariance=False) for ind in range(len(dehaze_list))
    ]

    return ssim_list


def print_log(epoch: int, num_epochs: int, one_epoch_time: str, train_psnr: float, val_psnr: float, val_ssim: float,
              category: str) -> None:
    print(
        f"({one_epoch_time:.0f}s) Epoch [{epoch}/{num_epochs}], Train_PSNR:{train_psnr:.2f}, Val_PSNR:{train_psnr:.2f}, Val_SSIM:{val_psnr:.4f}"
    )
    # write training log
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print(
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}s, Time_Cost: {one_epoch_time:.0f}s, Epoch: [{epoch}/{num_epochs}], Train_PSNR: {train_psnr:.2f}, Val_Image_PSNR: {val_psnr:.2f}, Val_Image_SSIM: {val_ssim:.4f}",
            file=f)


def adjust_learning_rate(optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler], epoch: int,
                         learning_rate: float, writer: torch.utils.tensorboard.SummaryWriter) -> float:  # type: ignore
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch > 0:  # and epoch < 16:
        if epoch % 2 == 0:
            learning_rate = scheduler.get_lr()[0]  # type: ignore
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                print('Learning rate sets to {}.'.format(param_group['lr']))
                scheduler.step()  # type: ignore
            writer.add_scalars('lr/train_lr_group', {
                'lr': learning_rate,
            }, epoch)
    return learning_rate


def poly_learning_decay(optimizer: Type[torch.optim.Optimizer], iter: int, total_epoch: int, loader_length: int,
                        writer: torch.utils.tensorboard.SummaryWriter) -> float:
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    max_iteration = total_epoch * loader_length
    learning_rate = optimizer.param_groups[0]['lr']
    learning_rate = learning_rate * (1 - iter / max_iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    writer.add_scalars('lr/train_lr_group', {
        'lr': learning_rate,
    }, iter)
    return learning_rate


def set_requires_grad(nets: list[torch.nn.Module], requires_grad: bool = False) -> None:
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def adjust_learning_rate_step(optimizer: torch.optim.Optimizer, epoch: int, num_epochs: int,
                              learning_rate: list[float]) -> float:
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    step = num_epochs // len(learning_rate)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate[epoch // step]
        print('Learning rate sets to {}.'.format(param_group['lr']))
    return learning_rate[epoch // step]
