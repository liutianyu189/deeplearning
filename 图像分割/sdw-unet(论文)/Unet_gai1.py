import torch
import torch.nn as nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c / squeeze_factor)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
        self.hs=nn.Hardsigmoid()
    def forward(self, x) :
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu6(scale)
        scale = self.fc2(scale)
        scale = self.hs(scale)
        return scale * x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class MBSE(nn.Module):
    def __init__(self, in_channel, out_channel,expand_factors):
        super(MBSE, self).__init__()
        layers = []
        hidden_channel = in_channel * expand_factors
        self.use_shortcut = (in_channel == out_channel)
        layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=1, groups=hidden_channel),
            SqueezeExcitation(hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
            if self.use_shortcut:
                return x + self.conv(x)
            else:
                return self.conv(x)
class FMBSE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FMBSE, self).__init__()
        layers = []
        hidden_channel=in_channel*1
        self.use_shortcut = (in_channel == out_channel)
        layers.extend([
            ConvBNReLU(in_channel, hidden_channel, stride=1,groups=hidden_channel),
            SqueezeExcitation(hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class conv_block0(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(conv_block0,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,groups=out_channels,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            )
    def forward(self,x):
        x=self.conv(x)
        return x
class conv_block1(nn.Module):
    def __init__(self,in_channels,out_channels,expand_factors):
        super(conv_block1,self).__init__()
        fmbse=FMBSE
        mbse = MBSE
        self.conv=nn.Sequential(
            mbse(in_channels,out_channels,expand_factors),
            fmbse(out_channels,out_channels),
            )
    def forward(self,x):
        x=self.conv(x)
        return x
class conv_block2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(conv_block2,self).__init__()
        fmbse=FMBSE
        self.conv=nn.Sequential(
            fmbse(in_channels,out_channels),
            )
    def forward(self,x):
        x=self.conv(x)
        return x

class upsamping(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsamping,self).__init__()
        self.up=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            )
    def forward(self,x):
        x=self.up(x)
        return x


class UNetg1(nn.Module):
    def __init__(self,in_channels=3,n_class=1):
        super(UNetg1,self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)  # 2×2
        self.conv1=conv_block0(in_channels,64)
        self.conv2=conv_block1(64,128,1)
        self.conv3=conv_block1(128,256,1)
        self.conv4=conv_block1(256,512,1)
        self.conv5=conv_block1(512,1024,1)
        self.upsamping5=upsamping(1024,512)
        self.upconv5=conv_block2(1024,512)
        self.upsamping4=upsamping(512,256)
        self.upconv4=conv_block2(512,256)
        self.upsamping3=upsamping(256,128)
        self.upconv3=conv_block2(256,128)
        self.upsamping2=upsamping(128,64)
        self.upconv2=conv_block2(128,64)
        self.upconv1=nn.Conv2d(64,n_class,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x1=self.conv1(x)  # [4, 64, 160, 160]
        x2=self.maxpool(x1)
        x2=self.conv2(x2)  # [4, 128, 80, 80]
        x3=self.maxpool(x2)
        x3=self.conv3(x3)  # [4, 256, 40, 40]
        x4=self.maxpool(x3)
        x4=self.conv4(x4)  # [4, 512, 20, 20]
        x5=self.maxpool(x4)
        x5=self.conv5(x5)  # [4, 1024, 10, 10]
        d5=self.upsamping5(x5)
        d5=torch.cat((x4,d5),dim=1)
        d5=self.upconv5(d5)  # [4, 512, 20, 20]
        d4=self.upsamping4(d5)
        d4=torch.cat((x3,d4),dim=1)
        d4=self.upconv4(d4)  # [4, 256, 40, 40]
        d3=self.upsamping3(d4)
        d3=torch.cat((x2,d3),dim=1)
        d3=self.upconv3(d3)  # [4, 128, 80, 80]
        d2=self.upsamping2(d3)
        d2=torch.cat((x1,d2),dim=1)
        d2=self.upconv2(d2)  # [4, 64, 160, 160]
        d1=self.upconv1(d2)   # [4, 2, 160, 160]
        return d1
# a=UNetg1()
# print(a)