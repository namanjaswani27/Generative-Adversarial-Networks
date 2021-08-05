from self_attention import Self_Attention
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import spectral_norm
from config import *

class Discriminator(nn.Module):

    def __init__(self, d_spectral_norm=False):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(DROPOUT)
        
        if d_spectral_norm:
            self.conv1 = spectral_norm(nn.Conv2d(
                in_channels = 1,
                out_channels = 64,
                kernel_size = 5,
                stride = 2,
                padding = 2,
                bias = True
            ))
        else:
            self.conv1 = nn.Conv2d(
                in_channels = 1,
                out_channels = 64,
                kernel_size = 5,
                stride = 2,
                padding = 2,
                bias = True
            )

        if d_spectral_norm:
            self.conv2 = spectral_norm(nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 5,
                stride = 2,
                padding = 2,
                bias = True
            ))
        else:
            self.conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 5,
            stride = 2,
            padding = 2,
            bias = True
            )

        if d_spectral_norm:
            self.linear = spectral_norm(nn.Linear(
                in_features = 128*7*7, out_features = 1, bias = True))
        else:
            self.linear = nn.Linear(
            in_features = 128*7*7, out_features = 1, bias = True)

        self.sigmoid = nn.Sigmoid()
        self.self_attention = Self_Attention(64) # input channels for last conv layer


    def forward(self, img):

        # x : [ batch_size, 64, 14, 14 ]
        x = self.conv1(img)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        if SELF_ATTENTION:
            x = self.self_attention(x)

        # x : [ batch_size, 128, 7, 7 ]
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # x : [ batch_size, 128*7*7 ]
        x = x.view(size = (-1, 128*7*7))

        # x : [ batch_size, 1 ]
        x = self.linear(x)
        output = self.sigmoid(x)

        return output
