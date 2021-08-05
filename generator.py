import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import spectral_norm
from config import *
from self_attention import *

class Generator(nn.Module):

    def __init__(self, latent_dim, g_channels, batch_norm=False, g_spectral_norm=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.batch_norm = batch_norm
        self.g_spectral_norm = g_spectral_norm

        if g_spectral_norm:
            self.linear = spectral_norm(nn.Linear(
                            in_features = latent_dim, out_features = g_channels*7*7, bias = False))
        else:
            self.linear = nn.Linear(
                in_features = latent_dim, out_features = g_channels*7*7, bias = False)
        
        self.bn1d = nn.BatchNorm1d(num_features = g_channels*7*7)

        self.leaky_relu = nn.LeakyReLU()

        if g_spectral_norm:
            self.conv1 = spectral_norm(nn.Conv2d(
                in_channels = g_channels,
                out_channels = 128,
                kernel_size = 5,
                stride = 1,
                padding = 2,
                bias = False
            ))
        else:
            self.conv1 = nn.Conv2d(
                in_channels = g_channels,
                out_channels = 128,
                kernel_size = 5,
                stride = 1,
                padding = 2,
                bias = False
            )


        self.bn2d1 = nn.BatchNorm2d(num_features = 128)

        if g_spectral_norm:
            self.convT1 = spectral_norm(nn.ConvTranspose2d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ))
        else:
            self.convT1 = nn.ConvTranspose2d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            )

        self.bn2d2 = nn.BatchNorm2d(num_features = 64)

        if g_spectral_norm:
            self.convT2 = spectral_norm(nn.ConvTranspose2d(
                in_channels = 64,
                out_channels = 1,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ))
        else:
            self.convT2 = nn.ConvTranspose2d(
                in_channels = 64,
                out_channels = 1,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            )

        self.tanh = nn.Tanh()
        self.self_attention = Self_Attention(64)   # input_channels for last conv layer


    def forward(self, latent_vector):
        
        # vec : [ batch_size, 256*7*7 ]
        x = self.linear(latent_vector)
        if self.batch_norm:
            x = self.bn1d(x)
        x = self.leaky_relu(x)

        # Pytorch takes images as [ C, H, W ] unlike TF
        # x : [ batch_size, 256, 7, 7 ]
        x = x.view(size = (-1, G_CHANNELS, 7, 7))

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn2d1(x)
        x = self.leaky_relu(x)

        x = self.convT1(x)
        if self.batch_norm:
            x = self.bn2d2(x)
        x = self.leaky_relu(x)
        # 256, 64, 14, 14
        if SELF_ATTENTION:
            x = self.self_attention(x)

        x = self.convT2(x)
        output = self.tanh(x)

        return output
        


        




