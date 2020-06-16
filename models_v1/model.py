import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DWT, IWT
from modules import shortcutblock, GCIWTResUp, GCWTResDown, GCRDB, ContextBlock2d

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, block=[2,3,3,3,5,7], device='cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
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
        self.layer5 = nn.Sequential(*_layer_5_dw)

        #upsample4
        self.layer4_up = GCIWTResUp(1024, ContextBlock2d, device)
        _layer_4_up = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        self.layer4_gcrdb = nn.Sequential(*_layer_4_dw)

        #upsample3
        self.layer4_up = GCIWTResUp(512, ContextBlock2d, device)
        _layer_4_up = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        self.layer4_gcrdb = nn.Sequential(*_layer_4_dw)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2, x2_dwt = self.layer1(x1)
        x3, x3_dwt = self.layer2(x2)
        x4, x4_dwt = self.layer3(x3)
        x5, x5_dwt = self.layer4(x4)
        x6 = self.layer5(x5)
        return x6

if __name__ == '__main__':
    x = torch.randn(1,4, 448, 448)
    net = Generator(4, 3)
    y = net(x)
    print(y.shape)