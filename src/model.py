# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
   def __init__(self, in_ch, out_ch):
       super().__init__()
       self.net = nn.Sequential(
           nn.Conv2d(in_ch, out_ch, 3, padding=1),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True),
           nn.Conv2d(out_ch, out_ch, 3, padding=1),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True),
       )


   def forward(self, x):
       return self.net(x)


class MultiScaleUNetAE(nn.Module):
   """
   Minimal U-Net Autoencoder for anomaly detection.
   Encoder -> bottleneck -> decoder with skip connections.
   """
   def __init__(self, in_channels=3, base=32):
       super().__init__()


       # Encoder
       self.enc1 = ConvBlock(in_channels, base)       # 256
       self.enc2 = ConvBlock(base, base*2)            # 128
       self.enc3 = ConvBlock(base*2, base*4)          # 64
       self.enc4 = ConvBlock(base*4, base*8)          # 32


       self.pool = nn.MaxPool2d(2)


       # Bottleneck
       self.bottleneck = ConvBlock(base*8, base*16)  # 16


       # Decoder
       self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
       self.dec4 = ConvBlock(base*16, base*8)


       self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
       self.dec3 = ConvBlock(base*8, base*4)


       self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
       self.dec2 = ConvBlock(base*4, base*2)


       self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
       self.dec1 = ConvBlock(base*2, base)


       # Output
       self.out = nn.Conv2d(base, in_channels, 1)


   def forward(self, x):
       # Encoder
       e1 = self.enc1(x)
       e2 = self.enc2(self.pool(e1))
       e3 = self.enc3(self.pool(e2))
       e4 = self.enc4(self.pool(e3))


       # Bottleneck
       b = self.bottleneck(self.pool(e4))


       # Decoder (skip connections)
       d4 = self.up4(b)
       d4 = self.dec4(torch.cat([d4, e4], dim=1))


       d3 = self.up3(d4)
       d3 = self.dec3(torch.cat([d3, e3], dim=1))


       d2 = self.up2(d3)
       d2 = self.dec2(torch.cat([d2, e2], dim=1))


       d1 = self.up1(d2)
       d1 = self.dec1(torch.cat([d1, e1], dim=1))


       out = torch.sigmoid(self.out(d1))
       return out




if __name__ == "__main__":
   # Quick sanity check
   model = MultiScaleUNetAE()
   x = torch.randn(2, 3, 256, 256)
   y = model(x)
   print("Output shape:", y.shape)