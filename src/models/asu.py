import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class ASU(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ASU, self).__init__()
        self.swin = SwinTransformer(
            patch_size=4, in_chans=in_channels, num_classes=out_channels, embed_dim=96, depths=[2, 2, 6, 2]
        )
        self.decoder = nn.Conv2d(96, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.swin(x)
        x = self.decoder(x)  # Output segmentation mask
        return x
