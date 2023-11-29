# -*- coding: utf-8 -*-
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import GCRDB, GCWTResDown, SE_net, ShortcutBlock


class BaseNet(nn.Module):
    def __init__(self, in_channels: int, block: list[int] = [2, 2, 2, 3, 4]) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        layer_1_dw: list[torch.nn.Module] = []
        for i in range(block[0]):
            layer_1_dw.append(GCRDB(64))
        layer_1_dw.append(GCWTResDown(64))
        self.layer1 = nn.Sequential(*layer_1_dw)

        layer_2_dw: list[torch.nn.Module] = []
        for i in range(block[1]):
            layer_2_dw.append(GCRDB(128))
        layer_2_dw.append(GCWTResDown(128))
        self.layer2 = nn.Sequential(*layer_2_dw)

        layer_3_dw: list[torch.nn.Module] = []
        for i in range(block[2]):
            layer_3_dw.append(GCRDB(256))
        layer_3_dw.append(GCWTResDown(256))
        self.layer3 = nn.Sequential(*layer_3_dw)

        layer_4_dw: list[torch.nn.Module] = []
        for i in range(block[3]):
            layer_4_dw.append(GCRDB(512))
        layer_4_dw.append(GCWTResDown(512))
        self.layer4 = nn.Sequential(*layer_4_dw)

        layer_5_dw: list[torch.nn.Module] = []
        for i in range(block[4]):
            layer_5_dw.append(GCRDB(1024))
        self.layer5 = nn.Sequential(*layer_5_dw)

        self.sc_x1 = ShortcutBlock(64)
        self.sc_x2 = ShortcutBlock(128)
        self.sc_x3 = ShortcutBlock(256)
        self.sc_x4 = ShortcutBlock(512)

        self.se1 = SE_net(64, 64)
        self.se2 = SE_net(128, 128)
        self.se3 = SE_net(256, 256)
        self.se4 = SE_net(512, 512)
        self.se5 = SE_net(1024, 1024)

        # Number of output channel is always 3 as we want to generate RGB images
        self.scale_5 = nn.Conv2d(1024, 3, kernel_size=3, padding=1)
        self.scale_4 = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        self.scale_3 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.scale_2 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.scale_1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def encoder_decoder_forward(self, input_image):
        x1 = self.conv1(input_image)

        x2, x2_dwt = self.layer1(self.se1(x1))
        x3, x3_dwt = self.layer2(self.se2(x2))
        x4, x4_dwt = self.layer3(self.se3(x3))
        x5, x5_dwt = self.layer4(self.se4(x4))
        x5_latent = self.layer5(self.se5(x5))

        x5_out = self.scale_5(x5_latent)
        x5_out = torch.sigmoid(x5_out)
        x4_up = self.layer4_up(x5_latent, x5_dwt) + self.sc_x4(x4)

        x4_out = self.scale_4(x4_up)
        x4_out = torch.sigmoid(x4_out)
        x3_up = self.layer3_up(x4_up, x4_dwt) + self.sc_x3(x3)

        x3_out = self.scale_3(x3_up)
        x3_out = torch.sigmoid(x3_out)
        x2_up = self.layer2_up(x3_up, x3_dwt) + self.sc_x2(x2)

        x2_out = self.scale_2(x2_up)
        x2_out = torch.sigmoid(x2_out)
        x1_up = self.layer1_up(x2_up, x2_dwt) + self.sc_x1(x1)

        return x1_up, x2_out, x3_out, x4_out, x5_out, x5_latent
