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
    def __init__(self, in_channel, out_channel, strides=1):
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
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel != out_channel or strides != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Identity()
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.conv_layer(x)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self, num_classes):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 2. define multiple residual blocks
        self.residual_layers = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 128, strides=2),
            ResBlock(128, 128),
            ResBlock(128, 256, strides=2),
            ResBlock(256, 256),
            ResBlock(256, 512, strides=2),
            ResBlock(512, 512),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # 3. define full-connected layer to classify
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.conv_layer(x)
        x = self.residual_layers(x)
        x = self.avg_pooling(x)
        # classification
        out = self.classifier(x)
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, cardinality=32, group_depth=4, strides=1):
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
        hidden_channels = cardinality * group_depth
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=strides, padding=1, groups=cardinality),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
        )
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel != out_channel or strides != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input 
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.conv_layer(x)
        shortcut = self.shortcut(x)
        out = nn.ReLU()(out + shortcut)
        return out


class ResNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 2. define multiple residual blocks
        self.resnext_layer = nn.Sequential(
            ResNextBlock(64, 128,strides=2),
            ResNextBlock(128, 128),
            ResNextBlock(128, 256, strides=2),
            ResNextBlock(256, 256),
            ResNextBlock(256, 512, strides=2, group_depth=8),
            ResNextBlock(512, 512, group_depth=8),
        )
        # 3. define full-connected layer to classify
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # 3. define full-connected layer to classify
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = self.avg_pooling(x)
        # classification
        out = self.classifier(x)
        return out

