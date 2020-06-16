import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from utils import DWT, IWT

class GCRDB(nn.Module):
    def __init__(self, in_channels, att_block, num_dense_layer=6, growth_rate=16):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(GCRDB, self).__init__()
        _in_channels = in_channels
        modules = []
        # modules.append(att_block(inplanes=_in_channels, planes=_in_channels))
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        modules.append(att_block(inplanes=_in_channels, planes=_in_channels))
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        self.final_att = att_block(inplanes=in_channels, planes=in_channels)
    def forward(self, x):
        out_rdb = self.residual_dense_layers(x)
        out_rdb = self.conv_1x1(out_rdb)
        out_rdb = self.final_att(out_rdb)
        out = out_rdb + x
        return out

class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class SE_net(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, attention=True):
        super().__init__()
        self.attention = attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_mid = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, padding=0)
        
        self.x_red = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):

        if self.attention is True:
            y = self.avg_pool(x)
            y = F.relu(self.conv_in(y))
            y = F.relu(self.conv_mid(y))
            y = torch.sigmoid(self.conv_out(y))
            x = self.x_red(x)
            return x * y
        else:
            return x

class ContextBlock2d(nn.Module):

    def __init__(self, inplanes=9, planes=32, pool='att', fusions=['channel_add'], ratio=4):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
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
            context_mask = self.softmax(context_mask)#softmax操作
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

    def forward(self, x):
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

class GCWTResDown(nn.Module):
    def __init__(self, in_channels, att_block, norm_layer):
        super().__init__()
        self.dwt = DWT()
        if norm_layer:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                        norm_layer(in_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                        norm_layer(in_channels),
                                        nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.att = att_block(in_channels*2, in_channels*2)
    def forward(self, x):
        stem = self.stem(x)
        xLL, dwt = self.dwt(x)
        res = self.conv1x1(xLL)
        out = torch.cat([stem, res], dim=1)
        out = self.att(out)
        return out, dwt

class GCIWTResUp(nn.Module):

    def __init__(self, in_channels, att_block, device, norm_layer=None):
        super().__init__()
        if norm_layer:
            self.stem = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                nn.Conv2d(in_channels, in_channels//2, kernel_size = 3, padding = 1),
                norm_layer(in_channels//2),
                nn.PReLU(),
                nn.Conv2d(in_channels//2, in_channels//2, kernel_size = 3, padding = 1),
                norm_layer(in_channels//2),
                nn.PReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                nn.Conv2d(in_channels, in_channels//2, kernel_size = 3, padding = 1),
                nn.PReLU(),
                nn.Conv2d(in_channels//2, in_channels//2, kernel_size = 3, padding = 1),
                nn.PReLU(),
            )
        self.conv1x1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0)
        self.att = att_block(in_channels//2, in_channels//2)
        self.iwt = IWT(device)

    def forward(self, x, x_dwt):
        stem = self.stem(x)
        x_iwt = self.iwt(x_dwt)
        out = torch.cat([stem, x_iwt], dim=1)
        out = self.conv1x1(out)
        out = self.att(out)
        return out

class shortcutblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.se = SE_net(in_channels, in_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.se(self.relu(self.conv2(self.relu(self.conv1(x)))))


if __name__ == '__main__':
    x = torch.randn(1, 64, 448, 448)
    net = GCRDB(64, ContextBlock2d)
    net2 = GCWTResDown(64, ContextBlock2d)
    net3 = GCIWTResUp(128, ContextBlock2d, 'cpu')
    y = net(x)
    y, dwt = net2(y)
    y = net3(y, dwt)
    print(y.shape)
    