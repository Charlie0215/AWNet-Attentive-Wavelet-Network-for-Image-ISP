# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import trainConfig
from dataloader_4channel import LoadData
from loss import ms_Loss
from models.model_4channel import AWNet
from utils import adjust_learning_rate_step, print_log, to_psnr, validation

np.random.seed(0)
torch.manual_seed(0)

# Dataset size
TRAIN_SIZE = 46839
TEST_SIZE = 1204
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train() -> None:
    device_ids = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Initialize loss and model
    loss = ms_Loss().to(device)
    net = AWNet(4, 3, block=[3, 3, 3, 4, 4]).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)  # type: ignore
    new_lr = trainConfig.learning_rate[0]

    # Reload
    if trainConfig.pretrain == True:
        net.load_state_dict(
            torch.load('{}/best_4channel.pkl'.format(trainConfig.save_best),
                       map_location=device)["model_state"])  # type: ignore
        print('weight loaded.')
    else:
        print('no weight loaded.')
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=new_lr, betas=(0.9, 0.999))

    # Dataloaders
    train_dataset = LoadData(trainConfig.data_dir, TRAIN_SIZE, dslr_scale=1, test=False)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=trainConfig.batch_size,
                              shuffle=True,
                              num_workers=32,
                              pin_memory=True,
                              drop_last=True)

    test_dataset = LoadData(trainConfig.data_dir, TEST_SIZE, dslr_scale=1, test=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=18,
                             pin_memory=True,
                             drop_last=False)

    print('Train loader length: {}'.format(len(train_loader)))

    pre_psnr, pre_ssim = validation(net, test_loader, device, save_tag=True)  # type: ignore
    print('previous PSNR: {:.4f}, previous ssim: {:.4f}'.format(pre_psnr, pre_ssim))
    iteration = 0
    for epoch in range(trainConfig.epoch):
        psnr_list = []
        start_time = time.time()
        if epoch > 0:
            new_lr = adjust_learning_rate_step(optimizer, epoch, trainConfig.epoch,
                                               trainConfig.learning_rate)  # type: ignore
        for batch_id, data in enumerate(train_loader):
            x, target, _ = data
            x = x.to(device)
            target = target.to(device)
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

            psnr_list.extend(to_psnr(pred[0], target))  # type: ignore

        train_psnr = sum(psnr_list) / len(psnr_list)
        state = {
            "model_state": net.state_dict(),
            "lr": new_lr,
        }
        print('saved checkpoint')
        torch.save(state, '{}/four_channel_epoch_{}.pkl'.format(trainConfig.checkpoints, epoch))

        one_epoch_time = time.time() - start_time
        print('time: {}, train psnr: {}'.format(one_epoch_time, train_psnr))
        val_psnr, val_ssim = validation(  # type: ignore
            net, test_loader, device, save_tag=True)
        print_log(
            epoch + 1,
            trainConfig.epoch,
            one_epoch_time,  # type: ignore
            train_psnr,
            val_psnr,
            val_ssim,
            'multi_loss')

        if val_psnr >= pre_psnr:
            state = {
                "model_state": net.state_dict(),
                "lr": new_lr,
            }

            print('saved best weight')
            torch.save(state, '{}/best_4channel.pkl'.format(trainConfig.save_best))
            pre_psnr = val_psnr


if __name__ == '__main__':
    train()
