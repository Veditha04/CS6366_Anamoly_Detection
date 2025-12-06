import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class BaselineAutoencoder(nn.Module):
    """
    Simple symmetric conv autoencoder (no skip connections).
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)  # /2

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)  # /4

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)  # /8

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels, base_channels)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x3 = self.enc3(x)
        x = self.pool3(x3)

        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = self.out_act(x)
        return x


class MultiScaleAutoencoder(nn.Module):
    """
    U-Net-style multi-scale autoencoder with skip connections.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4 * 2, base_channels * 4)  # concat with enc3

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2 * 2, base_channels * 2)  # concat with enc2

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)  # concat with enc1

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2_in = self.pool1(x1)

        x2 = self.enc2(x2_in)
        x3_in = self.pool2(x2)

        x3 = self.enc3(x3_in)
        x4_in = self.pool3(x3)

        x4 = self.bottleneck(x4_in)

        # Decoder with skips
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = self.out_act(x)
        return x
