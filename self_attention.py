import torch
import torch.nn as nn
import torch.nn.functional as F 
from config import *

class Self_Attention(nn.Module):
    '''
        Self Attention in GANs 
        ref: Self-Attention Generative Adversarial Networks [Zhang, GoodFellow et al.]
    '''

    def __init__(self, in_dim):
        super().__init__()

        self.f_conv = nn.Conv2d(
            in_channels = in_dim,
            out_channels = in_dim,
            kernel_size = 1
        )
        self.g_conv = nn.Conv2d(
            in_channels = in_dim,
            out_channels = in_dim,
            kernel_size = 1
        )
        self.h_conv = nn.Conv2d(
            in_channels = in_dim,
            out_channels = in_dim,
            kernel_size = 1
        )
        self.v_conv = nn.Conv2d(
            in_channels = in_dim,
            out_channels = in_dim,
            kernel_size = 1
        )

        # Intilizing by zero, as indicated in paper
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
        print('SELF ATT INITIALIZED')


    def forward(self, x):

        batch_size, C, height, width = x.shape
        # N : height x width to get pixel location

        # f_x : [ batch_size, N, C ]
        f_x = self.f_conv(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        # g_x : [ batch_size, C, N ]
        g_x = self.g_conv(x).view(batch_size, -1, height*width)
        # att_in : [ batch_size, N, N]
        att_in = torch.bmm(f_x, g_x) # batch matrix-matrix product

        # attention : [ batch_size, N, N]
        attention = self.softmax(att_in)
        # h_x : [ batch_size, C, N ]
        h_x = self.h_conv(x).view(batch_size, -1, height*width)

        out = torch.bmm(h_x, attention).view(batch_size, -1, height, width)
        # v_x = self.v_conv(out)

        # output_f_maps : [ batch_size, C, height, width ]
        output_f_maps = self.gamma * out + x

        return output_f_maps

