import torch
import torch.nn as nn
import torch.nn.functional as F
from models_v1.utils import DWT, IWT
from models_v1.modules import shortcutblock, GCIWTResUp, GCWTResDown, GCRDB, ContextBlock2d
import functools

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, block=[1,1,1,2,2]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        #layer1
        _layer_1_dw = []
        for i in range(block[0]):
            _layer_1_dw.append(GCRDB(64, ContextBlock2d))
        _layer_1_dw.append(GCWTResDown(64, ContextBlock2d, norm_layer=None))
        self.layer1 = nn.Sequential(*_layer_1_dw)

        #layer 2
        _layer_2_dw = []
        for i in range(block[1]):
            _layer_2_dw.append(GCRDB(128, ContextBlock2d))
        _layer_2_dw.append(GCWTResDown(128, ContextBlock2d, norm_layer=None))
        self.layer2 = nn.Sequential(*_layer_2_dw)

        #layer 3
        _layer_3_dw = []
        for i in range(block[2]):
            _layer_3_dw.append(GCRDB(256, ContextBlock2d))
        _layer_3_dw.append(GCWTResDown(256, ContextBlock2d, norm_layer=None))
        self.layer3 = nn.Sequential(*_layer_3_dw)

        #layer 4
        _layer_4_dw = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        _layer_4_dw.append(GCWTResDown(512, ContextBlock2d, norm_layer=None))
        self.layer4 = nn.Sequential(*_layer_4_dw)

        #layer 5
        _layer_5_dw = []
        for i in range(block[4]):
            _layer_5_dw.append(GCRDB(1024, ContextBlock2d))
        # _layer_5_dw.append(GCWTResDown(1024, ContextBlock2d, norm_layer=None))
        self.layer5 = nn.Sequential(*_layer_5_dw)

        #upsample4
        self.layer4_up = GCIWTResUp(1024, ContextBlock2d)
        _layer_4_up = []
        for i in range(block[3]):
            _layer_4_up.append(GCRDB(512, ContextBlock2d))
        self.layer4_gcrdb_up = nn.Sequential(*_layer_4_up)

        #upsample3
        self.layer3_up = GCIWTResUp(512, ContextBlock2d)
        _layer_3_up = []
        for i in range(block[2]):
            _layer_3_up.append(GCRDB(256, ContextBlock2d))
        self.layer3_gcrdb = nn.Sequential(*_layer_3_up)
        
        #upsample2
        self.layer2_up = GCIWTResUp(256, ContextBlock2d)
        _layer_2_up = []
        for i in range(block[1]):
            _layer_2_up.append(GCRDB(128, ContextBlock2d))
        self.layer2_gcrdb = nn.Sequential(*_layer_2_up)

        #upsample1
        self.layer1_up = GCIWTResUp(128, ContextBlock2d)
        _layer_1_up = []
        for i in range(block[1]):
            _layer_1_up.append(GCRDB(64, ContextBlock2d))
        self.layer1_gcrdb = nn.Sequential(*_layer_1_up)

        self.sc_x1 = shortcutblock(64)
        self.sc_x2 = shortcutblock(128)
        self.sc_x3 = shortcutblock(256)
        self.sc_x4 = shortcutblock(512)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2, x2_dwt = self.layer1(x1)
        x3, x3_dwt = self.layer2(x2)
        x4, x4_dwt = self.layer3(x3)
        x5, x5_dwt = self.layer4(x4)
        x5_latent = self.layer5(x5)
        x4_up = self.layer4_up(x5_latent, x5_dwt) + self.sc_x4(x4)
        x4_up = self.layer4_gcrdb_up(x4_up) 
        x3_up = self.layer3_up(x4_up, x4_dwt) + self.sc_x3(x3)
        x3_up = self.layer3_gcrdb(x3_up) 
        x2_up = self.layer2_up(x3_up, x3_dwt) + self.sc_x2(x2)
        x2_up = self.layer2_gcrdb(x2_up) 
        x1_up = self.layer1_up(x2_up, x2_dwt) + self.sc_x1(x1)
        x1_up = self.layer1_gcrdb(x1_up) 
        x1_up = self.final_conv(x1_up)
        return x1_up, x5_latent


class teacher_encoder(nn.Module):
    def __init__(self, in_channels, block=[1,1,1,2,2]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        #layer1
        _layer_1_dw = []
        for i in range(block[0]):
            _layer_1_dw.append(GCRDB(64, ContextBlock2d))
        _layer_1_dw.append(GCWTResDown(64, ContextBlock2d, norm_layer=None))
        self.layer1 = nn.Sequential(*_layer_1_dw)

        #layer 2
        _layer_2_dw = []
        for i in range(block[1]):
            _layer_2_dw.append(GCRDB(128, ContextBlock2d))
        _layer_2_dw.append(GCWTResDown(128, ContextBlock2d, norm_layer=None))
        self.layer2 = nn.Sequential(*_layer_2_dw)

        #layer 3
        _layer_3_dw = []
        for i in range(block[2]):
            _layer_3_dw.append(GCRDB(256, ContextBlock2d))
        _layer_3_dw.append(GCWTResDown(256, ContextBlock2d, norm_layer=None))
        self.layer3 = nn.Sequential(*_layer_3_dw)

        #layer 4
        _layer_4_dw = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        _layer_4_dw.append(GCWTResDown(512, ContextBlock2d, norm_layer=None))
        self.layer4 = nn.Sequential(*_layer_4_dw)

        #layer 5
        _layer_5_dw = []
        for i in range(block[4]):
            _layer_5_dw.append(GCRDB(1024, ContextBlock2d))
        # _layer_5_dw.append(GCWTResDown(1024, ContextBlock2d, norm_layer=None))
        self.layer5 = nn.Sequential(*_layer_5_dw)

    def forward(self, x):
        x1 = self.conv1(x)
        x2, x2_dwt = self.layer1(x1)
        x3, x3_dwt = self.layer2(x2)
        x4, x4_dwt = self.layer3(x3)
        x5, x5_dwt = self.layer4(x4)
        x5_latent = self.layer5(x5)
        return x1, x2, x2_dwt, x3, x3_dwt, x4, x4_dwt, x5, x5_dwt, x5_latent


class teacher_decoder(nn.Module):
    def __init__(self, out_channels, block=[1,1,1,2,2]):
        super().__init__()

        #upsample4
        self.layer4_up = GCIWTResUp(1024, ContextBlock2d)
        _layer_4_up = []
        for i in range(block[3]):
            _layer_4_up.append(GCRDB(512, ContextBlock2d))
        self.layer4_gcrdb_up = nn.Sequential(*_layer_4_up)

        #upsample3
        self.layer3_up = GCIWTResUp(512, ContextBlock2d)
        _layer_3_up = []
        for i in range(block[2]):
            _layer_3_up.append(GCRDB(256, ContextBlock2d))
        self.layer3_gcrdb = nn.Sequential(*_layer_3_up)
        
        #upsample2
        self.layer2_up = GCIWTResUp(256, ContextBlock2d)
        _layer_2_up = []
        for i in range(block[1]):
            _layer_2_up.append(GCRDB(128, ContextBlock2d))
        self.layer2_gcrdb = nn.Sequential(*_layer_2_up)

        #upsample1
        self.layer1_up = GCIWTResUp(128, ContextBlock2d)
        _layer_1_up = []
        for i in range(block[1]):
            _layer_1_up.append(GCRDB(64, ContextBlock2d))
        self.layer1_gcrdb = nn.Sequential(*_layer_1_up)

        self.sc_x1 = shortcutblock(64)
        self.sc_x2 = shortcutblock(128)
        self.sc_x3 = shortcutblock(256)
        self.sc_x4 = shortcutblock(512)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2, x2_dwt, x3, x3_dwt, x4, x4_dwt, x5, x5_dwt, x5_latent):
        x4_up = self.layer4_up(x5_latent, x5_dwt)
        x4_up = self.layer4_gcrdb_up(x4_up) + self.sc_x4(x4)
        x3_up = self.layer3_up(x4_up, x4_dwt)
        x3_up = self.layer3_gcrdb(x3_up) + self.sc_x3(x3)
        x2_up = self.layer2_up(x3_up, x3_dwt)
        x2_up = self.layer2_gcrdb(x2_up) + self.sc_x2(x2)
        x1_up = self.layer1_up(x2_up, x2_dwt)
        x1_up = self.layer1_gcrdb(x1_up) + self.sc_x1(x1)
        x1_up = self.final_conv(x1_up)
        # print(x1_up)
        return x1_up

class teacher(nn.Module):
    def __init__(self, path, is_train):
        self.is_train = is_train
        self.path = path
        if self.is_train:
            self.encoder = teacher_encoder(4)
            self.decoder = teacher_decoder(3)
        else:
            self.encoder = teacher_encoder(4)
            self.encoder.load_state_dict(torch.load('./weight/best_teacher.pkl')["model_state"])
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def save_model(self):
        state = {
            "model_state": self.encoder.state_dict(),
        }
        torch.save(state, '{}/matting_best.pkl'.format(self.path))
        
    def forward(self, x):
        if self.is_train:
            x1, x2, x2_dwt, x3, x3_dwt, x4, x4_dwt, x5, x5_dwt, x5_latent = self.encoder(x)
            out = self.decoder(x1, x2, x2_dwt, x3, x3_dwt, x4, x4_dwt, x5, x5_dwt, x5_latent)
            return out, x5_latent
        else:
            out = self.encoder(x)
            return out[-1]

        

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    x = torch.randn(1, 3, 448, 448)
    net = Generator(4, 3)
    dis = NLayerDiscriminator(3)
    y = dis(x)
    # y = net(x)
    print(y.shape)