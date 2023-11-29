# -*- coding: utf-8 -*-
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseNet
from models.modules import FourChannelLastUpsample, GCIWTResUpFourChannel


class AWNetFourChannel(BaseNet):
    def __init__(self, in_channels: int, block: list[int] = [2, 2, 2, 3, 4]):
        super().__init__(in_channels, block)
        self.layer4_up = GCIWTResUpFourChannel(2048)
        self.layer3_up = GCIWTResUpFourChannel(1024)
        self.layer2_up = GCIWTResUpFourChannel(512)
        self.layer1_up = GCIWTResUpFourChannel(256)

        self.last = FourChannelLastUpsample()

    def forward(self, x: torch.Tensor):
        x1_up, x2_out, x3_out, x4_out, x5_out, x5_latent = self.encoder_decoder_forward(x)
        x1_out = self.scale_1(x1_up)
        x1_out = torch.sigmoid(x1_out)
        out = self.last(x1_up)
        return (out, x1_out, x2_out, x3_out, x4_out, x5_out), x5_latent
