import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBlock import ConvBlock
from typing import List

class Decoder(nn.Module):
    def __init__(self, levels_in_encoder: int = 4, init_filters: int = 32, layers_per_block: int = 2,
                 conv_kernel_size: int = 3):
        """ Decoder (Expansive Path) for the U-Net architecture. """
        super().__init__()
        self.levels = levels_in_encoder - 1
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()


        for i in range(self.levels):
            in_ch_up = init_filters * (2**(levels_in_encoder - 1 - i))
            out_ch_up = in_ch_up // 2

            self.up_convs.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch_up,
                    out_channels=out_ch_up,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            )

            in_ch_conv = out_ch_up + out_ch_up
            out_ch_conv = out_ch_up

            self.conv_blocks.append(
                ConvBlock(in_ch_conv, out_ch_conv, layers_per_block=layers_per_block, kernel_size=conv_kernel_size)
            )

    # if you are careful about the input size, you can remove _crop_and_concat and just use torch.cat
    # but this is a bit more robust
    # and allows for more flexibility in the input size
    def _crop_and_concat(self, upsampled: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """ Crops skip connection to match upsampled tensor size and concatenates. """
        _, _, H_up, W_up = upsampled.shape
        _, _, H_skip, W_skip = skip_connection.shape

        h_diff = H_skip - H_up
        w_diff = W_skip - W_up

        if h_diff < 0 or w_diff < 0:
             padding = [w_diff // 2, w_diff - w_diff // 2,
                        h_diff // 2, h_diff - h_diff // 2]
             upsampled = F.pad(upsampled, [abs(p) for p in padding])
        else:
            skip_connection = skip_connection[:, :,
                                              h_diff // 2 : H_skip - (h_diff - h_diff // 2),
                                              w_diff // 2 : W_skip - (w_diff - w_diff // 2)]

        return torch.cat((upsampled, skip_connection), dim=1)


    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the decoder.
        Args:
            encoder_outputs (List[torch.Tensor]): List from Encoder (shallowest to deepest).
        Returns:
            torch.Tensor: Output of the final decoder layer.
        """
        x = encoder_outputs[-1]

        for i in range(self.levels):
            x = self.up_convs[i](x)

            skip_connection = encoder_outputs[-(i + 2)]

            x = self._crop_and_concat(x, skip_connection)
            x = self.conv_blocks[i](x)

        return x