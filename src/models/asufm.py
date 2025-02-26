import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class FocalModulation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FocalModulation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class ASUFM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ASUFM, self).__init__()
        self.swin = SwinTransformer(
            patch_size=4, in_chans=in_channels, num_classes=out_channels, embed_dim=96, depths=[2, 2, 6, 2]
        )
        self.fm = FocalModulation(in_channels, out_channels)

    def forward(self, x):
        x = self.swin(x)
        x = self.fm(x)  # Apply focal modulation
        return x
