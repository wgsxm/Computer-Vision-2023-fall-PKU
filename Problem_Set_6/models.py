import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 2. define full-connected layer to classify
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 4096),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.features(x)
        # classification
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
             # convolution
             # batch normalization
             # activate function
             # ......

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
             # 1x1 convolution
             # batch normalization
             # activate function
             # 3x3 convolution
             # ......
             # 1x1 convolution
             # ......

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        return out

