import torch
import torch.nn as nn
from typing import List
from ConvBlock import ConvBlock

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, levels: int = 4, init_filters: int = 32, layers_per_block: int = 2,
                 max_pool_kernel_size: int = 2, conv_kernel_size: int = 3):
        """ Encoder (Contracting Path) for the U-Net architecture. """
        super().__init__()
        self.levels = levels
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        out_ch = init_filters

        for i in range(levels):
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, layers_per_block=layers_per_block, kernel_size=conv_kernel_size)
            )
            if i < levels - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=max_pool_kernel_size))
            in_ch = out_ch
            out_ch *= 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for the encoder.
        Returns:
            List[torch.Tensor]: List of encoder outputs (skip connections + bottleneck).
                                Ordered from shallowest to deepest.
        """
        encoder_outputs = []
        for i in range(self.levels):
            x = self.conv_blocks[i](x)
            encoder_outputs.append(x)
            if i < self.levels - 1:
                x = self.pools[i](x)
        return encoder_outputs