import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
                in_channels = 1,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 0,
                bias = False
            )
        
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 0,
                bias = False
            )

        self.maxpool = nn.MaxPool2d(kernel_size = (2, 2))

        self.linear1 = nn.Linear(
                in_features = 12*12*64, out_features = 128, bias = False)
        self.linear2 = nn.Linear(
                in_features = 128, out_features = 10, bias = False)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)
    

    def forward(self, img):

        x = self.conv1(img)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = x.view(size = (-1, 12*12*64))

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.log_softmax(x)
    
        return x
