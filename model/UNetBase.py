import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from typing import List

class UNetBase(nn.Module):
    """ Base U-Net combining Encoder and Decoder. """
    def __init__(self, in_channels: int = 3, init_filters: int = 32, layers_per_block: int = 2,
                 levels: int = 4, kernel_size: int = 3, max_pool_kernel_size: int = 2):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels, init_filters=init_filters, layers_per_block=layers_per_block,
            levels=levels, conv_kernel_size=kernel_size, max_pool_kernel_size=max_pool_kernel_size
        )
        self.decoder = Decoder(
            levels_in_encoder=levels, init_filters=init_filters, layers_per_block=layers_per_block,
            conv_kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the U-Net base model. """
        encoder_outputs = self.encoder(x)
        decoder_output = self.decoder(encoder_outputs)
        return decoder_output