import torch.nn as nn
from torch.nn import functional as F
import torch
class Block1(nn.Module):
    def __init__(self,in_channel,hid_channel,out_channel,pad=1,dil=1,str=1):
        super(Block1, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channel,hid_channel , kernel_size=1,bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=3,padding=pad,dilation=dil,stride=str,bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.Conv2d(hid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,stride=str,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.act=nn.ReLU(inplace=True)
    def forward(self, x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.act(x1+x2)
        return x3

class Block2(nn.Module):
    def __init__(self,in_channel,hid_channel,out_channel,pad=1,dil=1):
        super(Block2, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channel,hid_channel , kernel_size=1,bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=3,padding=pad,dilation=dil,bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.Conv2d(hid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv2 = nn.Sequential()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.act(x1+x2)
        return x3

class ASPPconv(nn.Module):
    def __init__(self,inchannel,outchannel,kernel,pad=0,dil=1):
        super(ASPPconv,self).__init__()
        self.conv=nn.Sequential(
        nn.Conv2d(inchannel,outchannel,kernel,padding=pad,dilation=dil,bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU())
    def forward(self, x):
        x=self.conv(x)
        return x


class deeplabv3(nn.Module):
    def __init__(self,num_class=3):
        super(deeplabv3,self).__init__()
        layers = []
        layers.extend([nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        layers.extend(([Block1(64,64,256),
                        Block2(256,64,256),
                        Block2(256,64,256)]))
        layers.extend([Block1(256, 128, 512, str=2),
                       Block2(512, 128, 512),
                       Block2(512, 128, 512),
                       Block2(512, 128, 512)])
        layers.extend([Block1(512, 256, 1024),
                       Block2(1024, 256, 1024,dil=2,pad=2),
                       Block2(1024, 256, 1024,dil=2,pad=2),
                       Block2(1024, 256, 1024,dil=2,pad=2),
                       Block2(1024, 256, 1024,dil=2,pad=2),
                       Block2(1024, 256, 1024,dil=2,pad=2)])
        layers.extend([Block1(1024,512,2048,dil=2,pad=2),
                       Block2(2048,512,2048,pad=4,dil=4),
                       Block2(2048,512,2048,pad=4,dil=4)])
        self.conv0 = nn.Sequential(*layers)
        self.conv1=ASPPconv(2048,256,kernel=1)
        self.conv2=ASPPconv(2048,256,kernel=3,dil=12,pad=12)
        self.conv3=ASPPconv(2048, 256, kernel=3, dil=24, pad=24)
        self.conv4 =ASPPconv(2048, 256, kernel=3, dil=36, pad=36)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.conv5= ASPPconv(2048,256,kernel=1)
        self.endconv=nn.Sequential(
            nn.Conv2d(1280,256,kernel_size=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256,256,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,num_class,kernel_size=1,bias=False))

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2=self.conv2(x)
        x3 = self.conv3(x)
        x4=self.conv4(x)
        x5=F.interpolate(self.conv5(self.pool(x)),size=60,mode='bilinear', align_corners=False)
        x=torch.cat((x1,x2,x3,x4,x5),dim=1)
        x=self.endconv(x)
        x=F.interpolate(x,size=480,mode='bilinear', align_corners=False)
        return x


