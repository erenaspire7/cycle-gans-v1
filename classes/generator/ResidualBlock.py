import torch.nn as nn

from classes.generator.ConvolutionalBlock import ConvolutionalBlock


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        """
        In a residual block, the use of two ConvBlock instances with one having
        an activation function and the other not is a design choice that promotes
        the learning of residual information.

        The purpose of a residual block is to learn the residual mapping between
        the input and output of the block. The first ConvBlock in the sequence,
        which includes an activation function, helps in capturing and extracting
        important features from the input. The activation function introduces
        non-linearity, allowing the network to model complex relationships
        between the input and output.

        The second ConvBlock does not include an activation function.
        It mainly focuses on adjusting the dimensions (e.g., number of channels)
        of the features extracted by the first ConvBlock. The absence of an
        activation function in the second ConvBlock allows the block to learn
        the residual information. By directly adding the output of the second
        ConvBlock to the original input, the block learns to capture the
        residual features or changes needed to reach the desired output.

        (Information and explanation above generated by ChatGPT)
        """
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(
                channels, channels, add_activation=True, kernel_size=3, padding=1
            ),
            ConvolutionalBlock(
                channels, channels, add_activation=False, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        """
        This skip connection, achieved through the addition operation, helps
        in propagating gradients during training and alleviates the vanishing
        gradient problem. It also facilitates the flow of information from earlier
        layers to later layers, allowing the network to learn more effectively.

        (Information and explanation above generated by ChatGPT)
        """
        return x + self.block(x)
