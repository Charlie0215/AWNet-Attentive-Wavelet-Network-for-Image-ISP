# -*- coding: utf-8 -*-
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseNet
from models.modules import GCIWTResUpThreeChannel, PSPModule


class AWNetThreeChannel(BaseNet):
    def __init__(self, in_channels: int, block: list[int] = [2, 2, 2, 3, 4]) -> None:
        super().__init__(in_channels, block)
        self.layer4_up = GCIWTResUpThreeChannel(1024)
        self.layer3_up = GCIWTResUpThreeChannel(512)
        self.layer2_up = GCIWTResUpThreeChannel(256)
        self.layer1_up = GCIWTResUpThreeChannel(128)

        self.enhance = PSPModule(features=64, out_features=64, sizes=(1, 2, 3, 6))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        x1_up, x2_out, x3_out, x4_out, x5_out, x5_latent = self.encoder_decoder_forward(x)
        out = self.scale_1(x1_up)
        out = torch.sigmoid(out)

        return (out, x2_out, x3_out, x4_out, x5_out), x5_latent
