import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.layers import TemporalBlock, Chomp1d
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Temporal Convolutional Network (TCN) based classifier.

    Args:
        input_dim (int): feature dimension per time step (Conv1d in‑channels).
        num_channels (Sequence[int]): output channels for each residual block.
        kernel_size (int): size of the convolution kernel.
        dropout (float): dropout probability inside blocks.
        num_classes (int): output classes.
    """

    def __init__(self,
                 seq_len,
                 enc_in: int = 1,
                 num_channels: tuple = (128, 128, 128),
                 kernel_size: int = 7,
                 dropout: float = 0.2,
                 num_classes: int = 2):
        super().__init__()
        layers = []
        in_channels = enc_in
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_channels, out_channels,
                               kernel_size, stride=1,
                               dilation=dilation_size,
                               padding=padding,
                               dropout=dropout)
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim).
        Returns:
            logits tensor of shape (batch, num_classes).
        """
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        return x
