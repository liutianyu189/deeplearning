import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c,expand_c,squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.Hardswish()
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()
    def forward(self,x) :
        scale = nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x

class Fuse0(nn.Module):
    def __init__(self, in_channel):
        super(Fuse0, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(),
            nn.Identity()
        )
    def forward(self, x):
            return self.conv(x)

class Fuse1(nn.Module):
    def __init__(self, in_channel, expand,out_channel,drop_num=1.0,strid=1):
        super(Fuse1, self).__init__()
        hidden_channel=in_channel*expand
        self.use_shortcut = 0
        if in_channel==out_channel and  strid==1:
            self.use_shortcut=1
            self.dropout = DropPath(0.2*drop_num)
        else:self.dropout=nn.Identity()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=strid, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.Hardswish(),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            self.dropout
        )
    def forward(self, x):
        if self.use_shortcut == 1:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channel,expand, out_channel,drop_num=1.0,strid=1):
        super(InvertedResidual, self).__init__()
        self.use_shortcut=0
        hiden_channel=in_channel*expand
        if in_channel==out_channel and  strid==1:
            self.use_shortcut=1
            self.dropout = DropPath(0.2*drop_num)
        else:self.dropout=nn.Identity()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, hiden_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hiden_channel),
            nn.Hardswish(),
            nn.Conv2d(hiden_channel, hiden_channel, kernel_size=3, stride=strid, padding=1, groups=hiden_channel, bias=False),
            nn.BatchNorm2d(hiden_channel),
            nn.Hardswish(),
            SqueezeExcitation(in_channel,hiden_channel),
            nn.Conv2d(hiden_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
            self.dropout
        )
    def forward(self, x):
        if self.use_shortcut==1:
            return x+self.conv(x)
        else:
            return self.conv(x)

class EfficientNetv2(nn.Module):
    def __init__(self,num_classes=3):
        super(EfficientNetv2, self).__init__()
        block0 = Fuse0
        block1 = Fuse1
        block2 = InvertedResidual
        layers=[]
        layers.extend([
            nn.Conv2d(3, 24, kernel_size=3, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish()
        ])
        layers.extend([
            block0(24),
            block0(24),
            DropPath(1/34)
        ])
        layers.extend([
            block1(24,4,48,strid=2),
            block1(48, 4, 48,drop_num=2/34),
            block1(48, 4, 48,drop_num=3/34),
            block1(48, 4, 48,drop_num=4/34)
        ])
        layers.extend([
            block1(48,4,64,strid=2),
            block1(64, 4, 64,drop_num=5/34),
            block1(64, 4, 64,drop_num=6/34),
            block1(64, 4, 64,drop_num=7/34)
        ])
        layers.extend([
            block2(64,4,128,strid=2),
            block2(128, 4, 128,drop_num=8/34),
            block2(128, 4, 128,drop_num=9/34),
            block2(128, 4, 128,drop_num=10/34),
            block2(128, 4, 128,drop_num=11/34),
            block2(128, 4, 128,drop_num=12/34)
        ])
        layers.extend([
            block2(128,6,160),
            block2(160, 6, 160,drop_num=13/34),
            block2(160, 6, 160, drop_num=14/34),
            block2(160, 6, 160, drop_num=15/34),
            block2(160, 6, 160, drop_num=16/34),
            block2(160, 6, 160, drop_num=17/34),
            block2(160, 6, 160, drop_num=18/34),
            block2(160, 6, 160, drop_num=19/34),
            block2(160, 6, 160, drop_num=20/34)
        ])
        layers.extend([
            block2(160,6,256,strid=2),
            block2(256, 6, 256,drop_num=21/34),
            block2(256, 6, 256, drop_num=22/34),
            block2(256, 6, 256, drop_num=23/34),
            block2(256, 6, 256, drop_num=24/34),
            block2(256, 6, 256, drop_num=25/34),
            block2(256, 6, 256, drop_num=26/34),
            block2(256, 6, 256, drop_num=27/34),
            block2(256, 6, 256, drop_num=28/34),
            block2(256, 6, 256, drop_num=29/34),
            block2(256, 6, 256, drop_num=30/34),
            block2(256, 6, 256, drop_num=31/34),
            block2(256, 6, 256, drop_num=32/34),
            block2(256, 6, 256, drop_num=33/34),
            block2(256, 6, 256, drop_num=34/34),
        ])
        layers.extend([
            nn.Conv2d(256,1280,kernel_size=1,bias=False),
            nn.BatchNorm2d(1280),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
        ])
        self.conv=nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout2d(p=0.2, inplace=True),
            nn.Linear(1280,num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

