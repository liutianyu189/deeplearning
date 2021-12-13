from torch import nn
import torch

def _make_divisible(ch, divisor=8, min_ch=None):#通道设为8的倍数
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, kernel_size=1)
        self.hs=nn.Hardsigmoid()
    def forward(self, x):
        scale = nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = nn.functional.relu(scale)
        scale = self.fc2(scale)
        scale = self.hs(scale)
        return scale * x

class InvertedResidual0(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InvertedResidual0, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Identity()
        )
    def forward(self, x):
            return self.conv(x)

class InvertedResidual1(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand,pd=1,ks=3,re=0,se=0):
        super(InvertedResidual1, self).__init__()
        hidden_channel = expand
        self.use_shortcut = 0
        if stride == 1 and in_channel == out_channel:
            self.use_shortcut = 1
        layers=[]
        layers.extend([nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                       nn.BatchNorm2d(hidden_channel)])
        if re==0:
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Hardsigmoid())
        layers.extend([nn.Conv2d(hidden_channel, hidden_channel, kernel_size=ks, stride=stride, padding=pd, groups=hidden_channel, bias=False),
                       nn.BatchNorm2d(hidden_channel)])
        if re==0:
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Hardsigmoid())
        if se != 0:
            layers.append(SqueezeExcitation(hidden_channel))
        layers.extend([nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                      nn.BatchNorm2d(out_channel),
                      nn.Identity()]
                      )
        self.conv=nn.Sequential(*layers)
    def forward(self, x):
        if self.use_shortcut == 1:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetV3, self).__init__()
        block0=InvertedResidual0
        block1=InvertedResidual1
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            block0(16,16),
            #输入，输出，stride，expand
            block1(16, 24, 2, 64),
            block1(24, 24, 1, 72),
            block1(24, 40, 2, 72,ks=5,pd=2,se=1),
            block1(40, 40, 1, 120,ks=5,pd=2,se=1),
            block1(40, 40, 1, 120, ks=5,pd=2, se=1),
            block1(40, 80, 2, 240,re=1),
            block1(80, 80, 1, 200, re=1),
            block1(80, 80, 1, 184, re=1),
            block1(80, 80, 1, 184, re=1),
            block1(80, 112, 1, 480, re=1,se=1),
            block1(112, 112, 1, 672, re=1, se=1),
            block1(112, 160, 2, 672, ks=5,pd=2,re=1, se=1),
            block1(160, 160, 1, 960, ks=5,pd=2,re=1, se=1),
            block1(160, 160, 1, 960, ks=5,pd=2,re=1, se=1),
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2,inplace=True),
            nn.Linear(1280, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
        x = self.classifier(x)
        return x

