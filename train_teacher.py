# import pytorch dependencies
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import time
import threading
from tensorboardX import SummaryWriter
import numpy as np

from models_v1.model import Generator, PixelDiscriminator, teacher
from loss import Loss, ms_Loss
from dataloader_teacher import LoadData, LoadVisualData
from dataloader1 import DataLoaderX
from config import trainConfig
from utils import validation, adjust_learning_rate, writer_add_image, print_log, to_psnr, poly_learning_decay, adjust_learning_rate_step
import shutil
import os

np.random.seed(0)
torch.manual_seed(0)

# Dataset size
TRAIN_SIZE = 46839
TEST_SIZE = 1204

def train():
    log_path = './runs_teacher'
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)
    if torch.cuda.device_count() > 0:
        device_ids = [0]
        print('using device: {}'.format(device_ids))
    else: device_ids = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Initialize loss and model
    loss = ms_Loss().to(device)
    net = teacher(is_train=True, path=trainConfig.save_best, block=[3,3,3,4,4]).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    # Reload
    if trainConfig.pretrain == True:
        net.load_state_dict(torch.load('./weight/total_teacher_best.pkl')["model_state"])
        pre_lr = torch.load('./weight/total_teacher_best.pkl')["lr"]
        print('weight loaded.')
    else:
        print('no weight loaded.')
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # optimizer and scheduler
    new_lr = trainConfig.learning_rate[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=new_lr, betas=(0.9, 0.999))

    # Dataloaders
    train_dataset = LoadData(trainConfig.data_dir, TRAIN_SIZE, dslr_scale=2, test=False)
    train_loader = DataLoaderX(dataset=train_dataset, batch_size=trainConfig.batch_size, shuffle=True, num_workers=18,
                              pin_memory=True, drop_last=True)

    test_dataset = LoadData(trainConfig.data_dir, TEST_SIZE, dslr_scale=2, test=True)
    test_loader = DataLoaderX(dataset=test_dataset, batch_size=6, shuffle=False, num_workers=18,
                             pin_memory=True, drop_last=False)

    print('Train loader length: {}'.format(len(train_loader)))
    

    pre_psnr, pre_ssim = validation(net, test_loader, device, save_tag=True, mode='teacher')
    print(
        'previous PSNR: {:.4f}, previous ssim: {:.4f}'.format(pre_psnr, pre_ssim)
    )
    iteration = 0
    for epoch in range(trainConfig.epoch):
        # mse_list = []
        psnr_list = []
        start_time = time.time() 
        if epoch > 0:
            new_lr = adjust_learning_rate_step(optimizer, epoch, trainConfig.epoch, trainConfig.learning_rate) 
        for batch_id, data in enumerate(train_loader):
            x, target, _ = data
            # x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred, _ = net(x)

            optimizer.zero_grad()
            total_loss, losses = loss(pred, target)
            total_loss.backward()
            optimizer.step()

            iteration += 1
            if trainConfig.print_loss:
                print("epoch:{}/{} | Loss: {:.4f} ".format(epoch, trainConfig.epoch, total_loss.item()))
            if not (batch_id % 1000):
                print('Epoch:{0}, Iteration:{1}'.format(epoch, batch_id))

            writer.add_scalars('data/train_loss_group',
                               {'g_loss': total_loss.item(),
                                'perceptual_loss': losses[0].item(),
                                'l1': losses[1].item(),
                                'ssim': losses[2].item(),
                                #'tv': losses[3].item(),
                                }, iteration)

            psnr_list.extend(to_psnr(pred[0], target))

            if iteration % 100 == 0:
                threading.Thread(target=writer_add_image, args=('pred', writer, pred[0], iteration)).start()
                threading.Thread(target=writer_add_image, args=('target', writer, target, iteration)).start()
            del x, target

        train_psnr = sum(psnr_list) / len(psnr_list)        
        state = {
                "model_state": net.state_dict(),
                "lr": new_lr,
            }
        print('saved checkpoint')
        torch.save(state, '{}/teacher_epoch_{}.pkl'.format(trainConfig.checkpoints, epoch))
        
        one_epoch_time = time.time() - start_time
        print('time: {}, train psnr: {}'.format(one_epoch_time, train_psnr))
        val_psnr, val_ssim = validation(net, test_loader, device, save_tag=True, mode='teacher')
        print_log(epoch+1, trainConfig.epoch, one_epoch_time, train_psnr, val_psnr, val_ssim, 'multi_loss')

        writer.add_scalars('eval/validation_metrics_group',
                   {
                       'val psnr': val_psnr,
                        'val ssim': val_ssim, 
                   }, epoch)
        if val_psnr >= pre_psnr:
            state = {
                "model_state": net.state_dict(),
                "lr": new_lr,
            }
            net.module.save_model()
            print('saved best weight')
            torch.save(state, '{}/total_teacher_best.pkl'.format(trainConfig.save_best))
            pre_psnr = val_psnr

if __name__ == '__main__':
    train()