from dcn_v2 import dcn_v2_conv, DCNv2, DCN
import torch


if __name__ == '__main__':
    dcn = DCN(3, 64, kernel_size=(3, 3), stride=1,
              padding=1, deformable_groups=2).cuda(1)
    x1 = torch.randn((1,3,320,320)).cuda(1).contiguous()
    out = dcn(x1)
    print(out.shape)