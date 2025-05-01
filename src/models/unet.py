import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- Convolutional Block ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers_per_block: int = 2, kernel_size: int = 3):
        """
        Sequential Convolutional Block with BatchNorm and ReLU activation.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}")
        padding = kernel_size // 2

        layers = []
        # First layer: in_channels -> out_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Subsequent layers: out_channels -> out_channels
        for _ in range(layers_per_block - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the convolutional block. """
        return self.block(x)

# --- Encoder ---
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

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, levels_in_encoder: int = 4, init_filters: int = 32, layers_per_block: int = 2,
                 conv_kernel_size: int = 3):
        """ Decoder (Expansive Path) for the U-Net architecture. """
        super().__init__()
        self.levels = levels_in_encoder - 1
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        # Start from the filters at the bottleneck level and go up
        for i in range(self.levels):
            # Filters coming from the up-convolution (lower level in decoder)
            # Example: levels=4, init=32. Bottleneck=32*2^3=256.
            # i=0: UpConv(256 -> 128), ConvBlock(128+128 -> 128)  (using skip from level 2: 128 filters)
            # i=1: UpConv(128 -> 64),  ConvBlock(64+64 -> 64)    (using skip from level 1: 64 filters)
            # i=2: UpConv(64 -> 32),   ConvBlock(32+32 -> 32)     (using skip from level 0: 32 filters)
            in_ch_up = init_filters * (2**(levels_in_encoder - 1 - i))
            out_ch_up = in_ch_up // 2 # Halve the channels via transpose conv

            self.up_convs.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch_up,
                    out_channels=out_ch_up,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            )

            # Channels after concatenation: up-conv output + skip connection
            # Skip connection has same number of filters as up-conv output (out_ch_up)
            in_ch_conv = out_ch_up + out_ch_up
            out_ch_conv = out_ch_up

            self.conv_blocks.append(
                ConvBlock(in_ch_conv, out_ch_conv, layers_per_block=layers_per_block, kernel_size=conv_kernel_size)
            )

    def _crop_and_concat(self, upsampled: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """ Crops skip connection to match upsampled tensor size and concatenates. """
        # Note: ConvTranspose2d with kernel=2, stride=2 might not perfectly double size if input is odd.
                    # you can ensure that the input is even by resizing it before passing to the model
        # Padding might be necessary if skip connection is smaller, cropping if larger.
        # Using F.interpolate might be more robust, but cropping is standard U-Net.

        _, _, H_up, W_up = upsampled.shape
        _, _, H_skip, W_skip = skip_connection.shape

        # Calculate cropping margins (if skip is larger) or padding (if skip is smaller)
        h_diff = H_skip - H_up
        w_diff = W_skip - W_up

        if h_diff < 0 or w_diff < 0:
            # Need to pad the upsampled tensor (less common scenario with standard U-Net structure)
            # Padding format: (pad_left, pad_right, pad_top, pad_bottom)
            # Use absolute values for padding amounts
            padding = [abs(w_diff) // 2, abs(w_diff) - abs(w_diff) // 2, # Left, Right padding
                       abs(h_diff) // 2, abs(h_diff) - abs(h_diff) // 2] # Top, Bottom padding
            upsampled = F.pad(upsampled, padding)
            # print(f"Padding applied. Upsampled shape: {upsampled.shape}, Skip shape: {skip_connection.shape}")

        elif h_diff > 0 or w_diff > 0:
            # Need to crop the skip connection (standard U-Net scenario)
            # Cropping indices: [start_h : end_h, start_w : end_w]
            skip_connection = skip_connection[:, :,
                                              h_diff // 2 : H_skip - (h_diff - h_diff // 2),
                                              w_diff // 2 : W_skip - (w_diff - w_diff // 2)]
            # print(f"Cropping applied. Upsampled shape: {upsampled.shape}, Skip shape: {skip_connection.shape}")


        # Check final shapes before concat
        if upsampled.shape[2:] != skip_connection.shape[2:]:
             print(f"Warning: Mismatched spatial dimensions after crop/pad before concat.")
             print(f"Upsampled: {upsampled.shape}, Skip Connection: {skip_connection.shape}")
             # Attempt resize as fallback (might distort features)
             skip_connection = F.interpolate(skip_connection, size=upsampled.shape[2:], mode='bilinear', align_corners=False)
             print(f"Resized skip connection to: {skip_connection.shape}")

        # try to fucking imagine this
        return torch.cat((upsampled, skip_connection), dim=1)


    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the decoder.
        Args:
            encoder_outputs (List[torch.Tensor]): List from Encoder (shallowest to deepest).
        Returns:
            torch.Tensor: Output of the final decoder layer.
        """
        # Start with the bottleneck output (last element of encoder_outputs)
        x = encoder_outputs[-1]

        # Loop through decoder levels (from deepest up)
        for i in range(self.levels):
            # Apply up-convolution
            x = self.up_convs[i](x)

            # Get the corresponding skip connection from the encoder
            # Encoder outputs: [level0, level1, level2, bottleneck(level3)] (if levels=4)
            # Decoder steps (i): 0, 1, 2
            # i=0: Need skip from level 2. Index = -(0 + 2) = -2
            # i=1: Need skip from level 1. Index = -(1 + 2) = -3
            # i=2: Need skip from level 0. Index = -(2 + 2) = -4
            skip_connection = encoder_outputs[-(i + 2)]

            x = self._crop_and_concat(x, skip_connection)

            x = self.conv_blocks[i](x)

        return x

# --- U-Net Base (Encoder + Decoder) ---
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

# --- Full U-Net (Encoder + Decoder + 1x1Conv) ---
class UNet(nn.Module):
    """ Full U-Net model including the final classification layer. """
    def __init__(self, in_channels: int = 3, num_classes: int = 23, init_filters: int = 32,
                 layers_per_block: int = 2, levels: int = 4, kernel_size: int = 3,
                 max_pool_kernel_size: int = 2,
                 base_model: nn.Module = None, freeze_base_model: bool = False):
        super().__init__()

        if base_model is None:
            self.base_model = UNetBase(
                in_channels=in_channels, init_filters=init_filters,
                layers_per_block=layers_per_block, levels=levels, kernel_size=kernel_size,
                max_pool_kernel_size=max_pool_kernel_size
            )
        else:
            # If providing a custom base model, ensure its output channel count matches init_filters
            self.base_model = base_model
            print("Warning: Using provided base_model. Ensure its output channels match init_filters.")


        if freeze_base_model:
            print("Freezing base model parameters.")
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Final 1x1 convolution to map features to class scores
        # The input channels should match the output channels of the last decoder block
        self.final_conv = nn.Conv2d(in_channels=init_filters, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the full U-Net model. """
        base_output = self.base_model(x)
        output = self.final_conv(base_output)
        return output