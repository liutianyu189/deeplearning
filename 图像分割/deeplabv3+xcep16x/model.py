import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConv2d(nn.Module):
    def __init__(self, inc, outc, stride=1,dilation=1,pad=1):
        super(SepConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            inc, inc, kernel_size=3, stride=stride, padding=pad*dilation,
            dilation=dilation, groups=inc, bias=False)
        self.conv2=nn.Conv2d(inc, outc, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Block1(nn.Module):
    def __init__(self, inc, outc, stride=2, dilation=1):
        super(Block1, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(inc, outc, stride=1, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(outc, outc, stride=1, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(outc, outc, stride=stride, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        self.conv4=nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outc))
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        identity = self.conv4(identity)
        x = x + identity
        return F.relu(x, inplace=True)

class Block2(nn.Module):
    def __init__(self, inc, dilation=2):
        super(Block2, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(inc, inc, stride=1, dilation=dilation),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(inc, inc, stride=1, dilation=dilation),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(inc, inc, stride=1, dilation=dilation),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True))
        self.conv4=nn.Sequential()
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        identity = self.conv4(identity)
        x = x + identity
        return F.relu(x, inplace=True)

class Block3(nn.Module):
    def __init__(self, inc, outc, stride=1, dilation=2):
        super(Block3, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(inc, inc, stride=1, dilation=dilation),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(inc, outc, stride=1, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(outc, outc, stride=stride, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        self.conv4=nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outc))
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        identity = self.conv4(identity)
        x = x + identity
        return F.relu(x, inplace=True)

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

class deeplabv3_16xecp(nn.Module):
    def __init__(self,num_classes):
        super(deeplabv3_16xecp, self).__init__()
        # backbone
        self.conv1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2,bias=False,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(
            Block1(inc=64,outc=128))
        self.conv3=nn.Sequential(
            Block1(inc=128,outc=256),
            Block1(inc=256,outc=728,stride=2,dilation=1))#若是8x，stride=1,dilation=4
        layers=[]
        for i in range(1):
            layers.append(Block2(inc=728,dilation=2))#若是8x,dilation=4
        self.conv4=nn.Sequential(*layers)
        self.conv5=nn.Sequential(
            Block3(inc=728,outc=1024,stride=1,dilation=2),#若是8x,stride=1,dilation=4
            SepConv2d(inc=1024,outc=1536,dilation=2),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SepConv2d(inc=1536, outc=1536, dilation=2),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SepConv2d(inc=1536,outc=2048,dilation=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True))

        # ASPP
        self.ASconv1=ASPPconv(2048,256,kernel=1)
        self.ASconv2=ASPPconv(2048,256,kernel=3,dil=6,pad=6)
        self.ASconv3=ASPPconv(2048, 256, kernel=3, dil=12, pad=12)
        self.ASconv4 =ASPPconv(2048, 256, kernel=3, dil=18, pad=18)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.ASconv5= ASPPconv(2048,256,kernel=1)
        self.ASconvend=nn.Sequential(
            nn.Conv2d(1280,256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # low_level_feat
        self.conv6=nn.Sequential(
            nn.Conv2d(128,48,kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        # end
        self.conv7=nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


    def forward(self, x):
        #backbone
        x=self.conv1(x)
        x_l=self.conv2(x)
        x=self.conv3(x_l)
        x=self.conv4(x)
        x=self.conv5(x)
        
        #ASPP
        x1 = self.ASconv1(x)
        x2=self.ASconv2(x)
        x3 = self.ASconv3(x)
        x4=self.ASconv4(x)
        x5=F.interpolate(self.ASconv5(self.pool(x)),size=30,mode='bilinear', align_corners=True)
        x=torch.cat((x1,x2,x3,x4,x5),dim=1)
        x=self.ASconvend(x)
        x = F.interpolate(x, 120, mode='bilinear', align_corners=True)

        #low_level_feat
        x_l=self.conv6(x_l)

        #end
        x = torch.cat((x, x_l), dim=1)
        x=F.interpolate(self.conv7(x),480, mode='bilinear', align_corners=True)
        return x



# model=deeplabv3_16xecp(3)
# print(model)
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())
# params = list(model.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))
