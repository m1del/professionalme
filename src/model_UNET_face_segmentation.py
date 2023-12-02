import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        DoubleConvolution class performs a double convolution operation on the input tensor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            # First Convolution
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second Convolution
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConvolution module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after double convolution operation.
        """
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
                self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
        ):
            """
            Initializes the UNET model for face segmentation.

            Args:
                in_channels (int): Number of input channels. Default is 3.
                out_channels (int): Number of output channels. Default is 1.
                features (list): List of feature sizes for each layer. Default is [64, 128, 256, 512].
            """
            super(UNET, self).__init__()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            for feature in features:
                self.downs.append(DoubleConvolution(in_channels, feature))
                in_channels = feature

            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConvolution(feature*2, feature))

            self.bottleneck = DoubleConvolution(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        # Store skip connections for concatenation later
        skip_connections = []

        # Downsample path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsample path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Resize if necessary
            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:], antialias=True)

            # Concatenate skip connection and upsampled feature map
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Final convolution
        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    if preds.shape != x.shape:
        raise Exception(f"Expected output shape of {x.shape} but got {preds.shape}")
    else: print("Test passed!")
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()