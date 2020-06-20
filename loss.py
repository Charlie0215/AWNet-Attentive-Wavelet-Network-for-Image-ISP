import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
import numpy as np
import sys
from skimage import measure
import pytorch_ssim

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def forward(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor.expand_as(prediction)   
        loss = self.loss(prediction, target_tensor)
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, y, target, fake_label=None):
        total_loss, losses = self.image_restoration(y, target)
        if fake_label:
            total_loss += fake_label
        return total_loss, losses

    def image_restoration(self, pred, target):
        perceptual_loss = self.perceptual_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        ssim_loss = 1 - self.ssim_loss(pred, target)
        del pred, target
        Loss = perceptual_loss + l1 + ssim_loss

        return Loss, (perceptual_loss, l1, ssim_loss)

    def perceptual_loss(self, out_images, target_images):
        
        loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images)) / 3
        return loss

    def CharbonnierLoss(self, y, target, eps=1e-6):
        diff = y - target
        loss = torch.mean(torch.sqrt(diff * diff + eps))
        return loss

    def tv_loss(self, x, TVLoss_weight=1):
        def _tensor_size(t):
            return t.size()[1]*t.size()[2]*t.size()[3]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = _tensor_size(x[:,:,1:,:])
        count_w = _tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    



class ms_Loss(Loss):
    def __init__(self):
        super(ms_Loss, self).__init__()
    def forward(self, y, target):
        loss = 0
        total_l1 = 0
        total_perceptual=0
        total_ssim = 0
        # scale 1
        for i in range(len(y)):
            if i == 0:
                perceptual_loss = self.perceptual_loss(y[i], target)
                ssim_loss = 1 - self.ssim_loss(y[i], target)
                #l1 = F.smooth_l1_loss(y[i], target)
                l1 = self.CharbonnierLoss(y[i], target)
                tv_loss = self.tv_loss(y[i])
                loss += 0.25 * perceptual_loss + 0.05 * ssim_loss + l1 + 0.1 * tv_loss
                total_l1 += l1
                total_perceptual += perceptual_loss
                total_ssim += ssim_loss
            elif i == 1 or i == 2:
                h, w = y[i].size(2), y[i].size(3)
                target = F.interpolate(target, size=(h, w))
                perceptual_loss = self.perceptual_loss(y[i], target)
                # l1 = F.smooth_l1_loss(y[i], target)
                l1 = self.CharbonnierLoss(y[i], target)
                total_l1 += l1
                loss += perceptual_loss + l1 * 0.25
                total_perceptual += perceptual_loss
            else:
                h, w = y[i].size(2), y[i].size(3)
                target = F.interpolate(target, size=(h, w))
                l1 = self.CharbonnierLoss(y[i], target)
                total_l1 += l1
                loss += l1
                
        return loss, (total_perceptual, total_l1, total_ssim, tv_loss)

# laplacian loss
def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

if __name__ == '__main__':
    x1 = torch.randn(1,3,220,220)
    x2 = torch.randn(1,3,110,110)
    x3 = torch.randn(1,3,55,55)
    inp = [x1, x2, x3]
    y = torch.randn(1,3,220,220)
    loss = ms_Loss()
    l = loss(inp, y)