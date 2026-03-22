import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet


class LungAttentionUNet(nn.Module):
    """
    Attention U-Net wrapper using MONAI's AttentionUnet.

    Input:  (B, 3, 256, 256) — three consecutive CT slices (2.5D context)
    Output: (B, 1, 256, 256) — raw logits; apply sigmoid for probability mask
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.model = AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.1
        )

    def forward(self, x):
        return self.model(x)
