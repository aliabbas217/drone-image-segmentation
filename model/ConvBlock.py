import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers_per_block: int = 2, kernel_size: int = 3):
        """
        Sequential Convolutional Block with BatchNorm and ReLU activation.
        Args:
          in_channels (int): Number of input channels.
          out_channels (int): Number of output channels (filters).
          layers_per_block (int): Number of convolutional layers in the block.
          kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        # kerner size must be odd because of padding, checkout the original paper for more details
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}")
        padding = kernel_size // 2

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True)) 

        for _ in range(layers_per_block - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the convolutional block. """
        return self.block(x)