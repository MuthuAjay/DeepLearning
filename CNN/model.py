import torch
from torch import nn


class TinyVGG(nn.Module):

    def __init__(self,
                 input_shape: int,
                 output_classes: int,
                 hidden_units: int = 10) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3,
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      padding=0,
                      stride=1,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


class TinyVGGDynamic(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_classes: int,
                 hidden_unit: int = 10,
                 num_conv_blocks: int = 2,
                 kernel_size: int = 3) -> None:
        """
        TinyVGGDynamic1 model constructor.

        Args:
            input_shape (int): Number of input channels.
            output_classes (int): Number of output classes.
            hidden_unit (int): Number of hidden units in convolutional layers.
            num_conv_blocks (int): Number of convolutional blocks in the model.
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()

        self.in_channels = input_shape
        for _ in range(num_conv_blocks):
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_unit,
                          padding=0,
                          stride=1,
                          kernel_size=kernel_size),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_unit,
                          out_channels=hidden_unit,
                          padding=0,
                          stride=1,
                          kernel_size=kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                             stride=2)
            )
            self.conv_blocks.append(conv_block)
            input_shape = hidden_unit

        self.conv_output_size = self.get_conv_output_size(input_shape=self.in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.conv_output_size,
                      out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return self.classifier(x)

    def get_conv_output_size(self,
                             input_shape: int,
                             height: int = 64,
                             width: int = 64) -> int:
        """
        Calculate the output size of the convolutional blocks.

        Args:
            input_shape (int): Number of input channels.
            height (int): Height of the input tensor.
            width (int): Width of the input tensor.

        Returns:
            int: Size of the output tensor.
        """
        # It will find the output of the conv blocks
        # create dummy inputs
        # x = torch.rand(4, 10, 13, 13)
        # x.view(x.size(0), -1).shape
        # >> output: torch.Size([4, 1690])

        with torch.inference_mode():
            x = torch.rand(1, input_shape, height, width)
            for conv_block in self.conv_blocks:
                x = conv_block(x)
        return x.view(x.size(0), -1).size(1)
