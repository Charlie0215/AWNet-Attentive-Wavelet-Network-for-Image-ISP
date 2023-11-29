# -*- coding: utf-8 -*-
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import SE_net
from models.utils import DWT, IWT


class ContextBlock2d(nn.Module):
    def __init__(
        self,
        inplanes: int = 9,
        planes: int = 32,
        pool: str = "att",
        fusions: list[str] = ["channel_add"],
        ratio: int = 4,
    ) -> None:
        super(ContextBlock2d, self).__init__()
        assert pool in ["avg", "att"]
        assert all([f in ["channel_add", "channel_mul"] for f in fusions])
        assert len(fusions) > 0, "at least one fusion should be used"
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if "att" in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)  # context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if "channel_add" in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "channel_mul" in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        if self.pool == "att":
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class MakeDense(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int = 3) -> None:
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm_layer = nn.BatchNorm2d(growth_rate)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv(x))
        out = self.norm_layer(out)
        out = torch.cat((x, out), 1)
        return out


class GCWTResDown(nn.Module):
    def __init__(self, in_channels: int, norm_layer: torch.nn.Module = nn.BatchNorm2d) -> None:
        super().__init__()
        self.dwt = DWT()
        if norm_layer:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(in_channels),
                nn.PReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(in_channels),
                nn.PReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
            )
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_down = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        # self.att = att_block(in_channels * 2, in_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        xLL, dwt = self.dwt(x)
        res = self.conv1x1(xLL)
        out = torch.cat([stem, res], dim=1)
        # out = self.att(out)
        return out, dwt


class GCIWTResUp(nn.Module):
    def __init__(self, in_channels: int, norm_layer: Optional[Type[torch.nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer:
            self.stem = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                norm_layer(in_channels // 2),
                nn.PReLU(),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
                norm_layer(in_channels // 2),
                nn.PReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
                nn.PReLU(),
            )

        self.pre_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, padding=0)
        self.prelu = nn.PReLU()
        self.conv1x1 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, padding=0)
        self.iwt = IWT()

    def forward(self, x: torch.Tensor, x_dwt: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        x_dwt = self.prelu(self.pre_conv(x_dwt))
        x_iwt = self.iwt(x_dwt)
        x_iwt = self.conv1x1(x_iwt)
        out = torch.cat([stem, x_iwt], dim=1)
        return out


class shortcutblock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.se = SE_net(in_channels, in_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.relu(self.conv2(self.relu(self.conv1(x)))))
