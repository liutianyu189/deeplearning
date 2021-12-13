from torch import nn
import torch

def _make_divisible(ch, divisor=8, min_ch=None):#通道设为8的倍数
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class InvertedResidual0(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InvertedResidual0, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
    def forward(self, x):
            return self.conv(x)

class InvertedResidual1(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual1, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = 0
        if stride == 1 and in_channel == out_channel:
            self.use_shortcut = 1
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=hidden_channel, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
    def forward(self, x):
        if self.use_shortcut == 1:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3,a=1):
        super(MobileNetV2, self).__init__()
        block0=InvertedResidual0
        block1=InvertedResidual1
        self.conv = nn.Sequential(
            nn.Conv2d(3, _make_divisible(32 * a, 8), kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(_make_divisible(32 * a, 8)),
            nn.ReLU6(inplace=True),
            block0(_make_divisible(32 * a, 8),16),
            block1(16, 24, 2, 6),
            block1(24, 24, 1, 6),
            block1(24, 32, 2, 6),
            block1(32, 32, 1, 6),
            block1(32, 32, 1, 6),
            block1(32, 64, 2, 6),
            block1(64, 64, 1, 6),
            block1(64, 64, 1, 6),
            block1(64, 64, 1, 6),
            block1(64, 96, 1, 6),
            block1(96, 96, 1, 6),
            block1(96, 96, 1, 6),
            block1(96, 160, 2, 6),
            block1(160, 160, 1, 6),
            block1(160, 160, 1, 6),
            block1(160, 320, 1, 6),
            nn.Conv2d(320, _make_divisible(1280 * a, 8), kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(_make_divisible(1280 * a, 8)),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(_make_divisible(1280 * a), num_classes)
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


