import torch
import torch.nn as nn
import math
from torchinfo import summary
import torch.nn.functional as F

class Conv2dSame(torch.nn.Conv2d):
    """Padded 2D convolution using 'same' approach."""
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class FeatureMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Conv2dSame(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=2
        )
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)
        
        return x

class FeatureMapUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1
        )
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)
        
        return x

class VAELatent(nn.Module):
    def __init__(self, in_channels, num_latent):
        super().__init__()
        
        self.kl = 0
        
        # in conv stuff
        self.conv_in = nn.Conv2d(
            in_channels,
            16,
            kernel_size=1,
            stride=1
        )
        self.bnorm_in = nn.BatchNorm2d(16)
        self.relu_in = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        
        # latent sampling
        self.mu = nn.Linear(1024, num_latent)
        self.sigma = nn.Linear(1024, num_latent)
        self.dropout_mu = nn.Dropout(p=0.1)
        self.dropout_sigma = nn.Dropout(p=0.1)
        self.dense_out = nn.Linear(num_latent, 1024)
        
        # out conv stuff
        self.unflatten = nn.Unflatten(-1, (16, 8, 8))
        self.conv_out = nn.ConvTranspose2d(
            16,
            in_channels,
            kernel_size=1,
            stride=1
        )
        self.bnorm_out = nn.BatchNorm2d(in_channels)
        self.relu_out = nn.LeakyReLU()
    
    def forward(self, x):
        # one more convolution
        x = self.conv_in(x)
        x = self.bnorm_in(x)
        x = self.relu_in(x)
        x = self.flatten(x)
        
        # Variational latent
        mu = self.mu(x)
        mu = self.dropout_mu(mu)
        
        sigma = self.sigma(x)
        sigma = self.dropout_sigma(sigma)
        
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        x = mu + eps * std
        
        # get output back
        x = self.dense_out(x)
        x = self.unflatten(x)
        x = self.conv_out(x)
        x = self.bnorm_out(x)
#         x = self.relu_out(x)
        
        return x #, mu, sigma

class VAE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = [32, 32, 32, 32, 32]
        # self.channels = [128, 64, 32, 16, 8]

        self.down_layers = nn.Sequential()
        for i in self.channels:
            self.down_layers.append(FeatureMap(self.in_channels, i))
            self.in_channels = i
        
        self.latent = VAELatent(self.in_channels, 512)
        
        self.up_layers = nn.Sequential()
        for i in self.channels[::-1]:
            self.up_layers.append(FeatureMapUp(self.in_channels, i))
            self.in_channels = i
        
        self.last_layer = nn.ConvTranspose2d(
            self.in_channels,
            in_channels,
            1,
            1
        )
        
    def forward(self, x):
        x = self.down_layers(x)
        x = self.latent(x)
        x = self.up_layers(x)
        x = self.last_layer(x)
        
        return x #, mu, sigma
    
class Classifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
        # self.channels = [32, 32, 32, 32, 32]
        self.channels = [16, 16, 16, 16, 16]

        self.down_layers = nn.Sequential()
        for i in self.channels:
            self.down_layers.append(FeatureMap(self.in_channels, i))
            self.in_channels = i


class ResnetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.padding = 'valid'
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ## Encoder
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        
        return x


class UNetBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        self.resnetblock = ResnetEncoder(in_channels, out_channels)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.resnetblock(x)
        
        return x

    
class UNetBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.transpose = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.conv = ResnetEncoder(in_channels, out_channels)
        
    def forward(self, x, x_bypass):
        x = self.transpose(x)
        
        x_diff = x_bypass.size()[2] - x.size()[2]
        y_diff = x_bypass.size()[3] - x.size()[3]
        
        
        x = F.pad(
            x,
            [x_diff // 2, x_diff - x_diff // 2, y_diff // 2, y_diff - y_diff // 2]
        )
        
        x = torch.cat([x_bypass, x], dim=1)
        x = self.conv(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.input = ResnetEncoder(in_channels, 32)
        self.encoder1 = UNetBlockDown(32, 64)
        self.encoder2 = UNetBlockDown(64, 128)
        self.encoder3 = UNetBlockDown(128, 256)
        self.encoder4 = UNetBlockDown(256, 512)
        
        self.decoder1 = UNetBlockUp(512, 256)
        self.decoder2 = UNetBlockUp(256, 128)
        self.decoder3 = UNetBlockUp(128, 64)
        self.decoder4 = UNetBlockUp(64, 32)
        self.output = nn.Conv2d(32, in_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.input(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x5 = self.decoder1(x4, x3)
        x6 = self.decoder2(x5, x2)
        x7 = self.decoder3(x6, x1)
        x8 = self.decoder4(x7, x)
        x8 = self.output(x8)
        
        return x8
    

class MiniUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.input = ResnetEncoder(in_channels, 32)
        self.encoder1 = UNetBlockDown(32, 64)
        self.encoder2 = UNetBlockDown(64, 128)
#         self.encoder3 = UNetBlockDown(128, 256)
#         self.encoder4 = UNetBlockDown(256, 512)
        
#         self.decoder1 = UNetBlockUp(512, 256)
#         self.decoder2 = UNetBlockUp(256, 128)
        self.decoder1 = UNetBlockUp(128, 64)
        self.decoder2 = UNetBlockUp(64, 32)
        self.output = nn.Conv2d(32, in_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.input(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        # x3 = self.encoder3(x2)
        # x4 = self.encoder4(x3)
        
        x3 = self.decoder1(x2, x1)
        x4 = self.decoder2(x3, x)
        # x7 = self.decoder3(x6, x1)
        # x8 = self.decoder4(x7, x)
        x5 = self.output(x4)
        
        return x5


class MinierUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.input = ResnetEncoder(in_channels, 32)
        self.encoder1 = UNetBlockDown(32, 64)
        # self.encoder2 = UNetBlockDown(64, 128)
#         self.encoder3 = UNetBlockDown(128, 256)
#         self.encoder4 = UNetBlockDown(256, 512)
        
#         self.decoder1 = UNetBlockUp(512, 256)
#         self.decoder2 = UNetBlockUp(256, 128)
        # self.decoder1 = UNetBlockUp(128, 64)
        self.decoder2 = UNetBlockUp(64, 32)
        self.output = nn.Conv2d(32, in_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.input(x)
        x1 = self.encoder1(x)
        # x2 = self.encoder2(x1)
        # x3 = self.encoder3(x2)
        # x4 = self.encoder4(x3)
        
        # x3 = self.decoder1(x2, x1)
        x4 = self.decoder2(x1, x)
        # x7 = self.decoder3(x6, x1)
        # x8 = self.decoder4(x7, x)
        x5 = self.output(x4)
        
        return x5