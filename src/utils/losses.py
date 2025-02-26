import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def forward(self, inputs, targets, alpha=0.8, gamma=2):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()

class TverskyLoss(nn.Module):
    def forward(self, inputs, targets, alpha=0.7, beta=0.3, smooth=1):
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

class IoULoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        return 1 - (intersection + smooth) / (union + smooth)

class BCELoss(nn.Module):
    def forward(self, inputs, targets):
        return F.binary_cross_entropy(inputs, targets)