# -*- coding: utf-8 -*-
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import trainConfig
from data.dataloader_4channel import LoadData
from loss import ms_Loss
from models.model_4channel import AWNet
from params import PipelineParams
from utils import adjust_learning_rate_step, get_log, in_training_validation, load_yaml_config, setup_logging, to_psnr

np.random.seed(0)
torch.manual_seed(0)

# Dataset size
TRAIN_SIZE = 46839
TEST_SIZE = 1204
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(params: PipelineParams) -> None:
    ########## Setup logging utilities ##########
    log_dir, checkpoint_saving_dir, _ = setup_logging(Path(params.training_params.root_log_dir),
                                                      params.training_params.experiment_name)
    device_ids = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("CUDA visible devices: " + str(torch.cuda.device_count()))
    logging.info("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    ########## Initialize criterion and model ##########
    loss = ms_Loss().to(device)
    net = AWNet(in_channels=params.awnet_model_params.input_num_channels,
                block=params.awnet_model_params.num_gcrdb).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)  # type: ignore

    ########## Initialize model weight ##########
    if trainConfig.pretrain == True:
        net.load_state_dict(torch.load(log_dir / "best-model.pkl", map_location=device)["model_state"])  # type: ignore
        logging.info('Pretrained weight loaded.')
    else:
        logging.info('No pretrained weight loaded.')
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info("Total number of params: {}".format(pytorch_total_params))

    ########## Setup optimizer and scheduler ##########
    current_lr = params.training_params.learning_rate_milestones[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=current_lr, betas=(0.9, 0.999))

    ########## Dataset and dataloader setup ##########
    train_dataset = LoadData(Path(params.dataset_params.train_dataset_dir),
                             TRAIN_SIZE,
                             dslr_scale=params.dataset_params.resized_size,
                             test=False)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=params.dataloader_params.train.batch_size,
                              shuffle=params.dataloader_params.train.shuffle,
                              num_workers=params.dataloader_params.train.num_workers,
                              pin_memory=params.dataloader_params.train.pin_memory,
                              drop_last=True)

    test_dataset = LoadData(Path(params.dataset_params.val_dataset_dir),
                            TEST_SIZE,
                            dslr_scale=params.dataset_params.resized_size,
                            test=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=params.dataloader_params.val.batch_size,
                             shuffle=params.dataloader_params.val.shuffle,
                             num_workers=params.dataloader_params.val.num_workers,
                             pin_memory=params.dataloader_params.val.pin_memory,
                             drop_last=False)

    logging.info('Train loader length: {}'.format(len(train_loader)))

    ########## Compute baseline psnr and ssim ##########
    pre_psnr, pre_ssim = in_training_validation(net, test_loader, device, save_tag=True,
                                                log_dir=log_dir)  # type: ignore
    logging.info('previous PSNR: {:.4f}, previous ssim: {:.4f}'.format(pre_psnr, pre_ssim))
    for epoch in range(trainConfig.epoch):
        psnr_list = []
        start_time = time.time()
        if epoch > 0:
            current_lr = adjust_learning_rate_step(optimizer, epoch, trainConfig.epoch,
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

            if params.training_params.print_loss and batch_id % 1000 == 0:
                logging.info(
                    f'Epoch:{epoch}/{params.training_params.num_epoch} | Iteration:{batch_id} | Loss: {total_loss.item():.4f} '
                )

            psnr_list.extend(to_psnr(pred[0], target))  # type: ignore

        train_psnr = sum(psnr_list) / len(psnr_list)
        state = {
            "model_state": net.state_dict(),
            "lr": current_lr,
        }
        torch.save(state, checkpoint_saving_dir / f"epoch_{epoch}.pkl")
        logging.info('Saved checkpoint.')

        one_epoch_time = time.time() - start_time
        logging.info(f"time: {one_epoch_time}, train psnr: {train_psnr}")
        val_psnr, val_ssim = in_training_validation(  # type: ignore
            net, test_loader, device, save_tag=True)
        get_log(
            epoch + 1,
            trainConfig.epoch,
            one_epoch_time,  # type: ignore
            train_psnr,
            val_psnr,
            val_ssim)

        # Update the best psnr, note that we use psnr to choose our best model
        if val_psnr >= pre_psnr:
            state = {
                "model_state": net.state_dict(),
                "lr": current_lr,
            }

            logging.info('Saved best weight.')
            torch.save(state, log_dir / "best_3channel.pkl")
            pre_psnr = val_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for training 3-channel model.")
    parser.add_argument("-c", "--config-file", type=Path, help="Path to the training config.")
    args = parser.parse_args()
    training_config: PipelineParams = load_yaml_config(PipelineParams, args.config_file)
    train(training_config)
