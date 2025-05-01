import torch.nn as nn
import torch
from UNetBase import UNetBase

class UNet(nn.Module):
    """ Full U-Net model including the final classification layer. """
    def __init__(self, in_channels: int = 3, num_classes: int = 23, init_filters: int = 32,
                 layers_per_block: int = 2, levels: int = 4, kernel_size: int = 3,
                 base_model: nn.Module = None, freeze_base_model: bool = False):
        super().__init__()

        if base_model is None:
            self.base_model = UNetBase(
                in_channels=in_channels, init_filters=init_filters,
                layers_per_block=layers_per_block, levels=levels, kernel_size=kernel_size
            )
        else:
            self.base_model = base_model

        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.final_conv = nn.Conv2d(in_channels=init_filters, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the full U-Net model. """
        base_output = self.base_model(x)
        output = self.final_conv(base_output)
        return output