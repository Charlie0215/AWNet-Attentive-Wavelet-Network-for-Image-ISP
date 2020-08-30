import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
import pytorch_ssim


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

        loss = self.mse_loss(
            self.loss_network(out_images),
            self.loss_network(target_images)) / 3
        return loss

    def CharbonnierLoss(self, y, target, eps=1e-6):
        diff = y - target
        loss = torch.mean(torch.sqrt(diff * diff + eps))
        return loss

    def tv_loss(self, x, TVLoss_weight=1):
        def _tensor_size(t):
            return t.size()[1] * t.size()[2] * t.size()[3]

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = _tensor_size(x[:, :, 1:, :])
        count_w = _tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return TVLoss_weight * 2 * (
            h_tv / count_h + w_tv / count_w) / batch_size


class ms_Loss(Loss):
    def __init__(self):
        super(ms_Loss, self).__init__()

    def forward(self, y, target, texture_img=None):
        loss = 0
        total_l1 = 0
        total_perceptual = 0
        total_ssim = 0
        # scale 1
        if texture_img:
            l1 = self.CharbonnierLoss(texture_img, target) * 0.25
            total_l1 += l1
            loss += l1
        for i in range(len(y)):
            if i == 0:
                perceptual_loss = self.perceptual_loss(y[i], target)
                ssim_loss = 1 - self.ssim_loss(y[i], target)
                l1 = self.CharbonnierLoss(y[i], target)
                loss += 0.05 * ssim_loss + l1 + 0.25 * perceptual_loss
                total_l1 += l1
                total_perceptual += perceptual_loss
                total_ssim += ssim_loss
            elif i == 1 or i == 2:
                h, w = y[i].size(2), y[i].size(3)
                target = F.interpolate(target, size=(h, w))
                perceptual_loss = self.perceptual_loss(y[i], target)
                l1 = F.smooth_l1_loss(y[i], target)
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

        return loss, (total_perceptual, total_l1, total_ssim)
