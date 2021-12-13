import torch.nn as nn
import torch.nn.functional as F
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

class fcn(nn.Module):
    def __init__(self,num_class):
        super(fcn,self).__init__()
        layers=[]
        layers.extend([nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        layers.extend([Block1(64,64,256),
                       Block2(256,64,256),
                       Block2(256, 64, 256)
                       ])
        layers.extend([Block1(256, 128,512,str=2),
                       Block2(512, 128, 512),
                       Block2(512, 128, 512),
                       Block2(512, 128, 512)
                       ])
        layers.extend([Block1(512,256,1024),
                       Block2(1024,256,1024,pad=2,dil=2),
                       Block2(1024,256,1024,pad=2,dil=2),
                       Block2(1024,256,1024,pad=2,dil=2),
                       Block2(1024,256,1024,pad=2,dil=2),
                       Block2(1024,256,1024,pad=2,dil=2)
        ])
        layers.extend([Block1(1024,512,2048,pad=2,dil=2),
                       Block2(2048,512,2048,pad=4,dil=4),
                       Block2(2048, 512, 2048, pad=4, dil=4)
        ])
        layers.extend([nn.Conv2d(2048,512,kernel_size=3,padding=1,bias=False),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.1),
                       nn.Conv2d(512,num_class,kernel_size=1)
        ])
        self.conv=nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv(x)
        x = F.interpolate(x, size=512, mode='bilinear', align_corners=False)
        return x



