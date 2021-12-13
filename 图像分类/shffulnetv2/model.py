from torch import nn
import torch

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class InvertedResidual0(nn.Module):
    def __init__(self, in_channel):
        super(InvertedResidual0, self).__init__()
        split_channel=in_channel//2
        self.conv1=nn.Sequential()
        self.conv2=nn.Sequential(
            nn.Conv2d(split_channel, split_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(split_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(split_channel, split_channel, kernel_size=3, stride=1, padding=1, groups=split_channel, bias=False),
            nn.BatchNorm2d(split_channel),
            nn.Conv2d(split_channel, split_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(split_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.conv1(x1), self.conv2(x2)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class InvertedResidual1(nn.Module):
    def __init__(self, in_channel,hid_channel):
        super(InvertedResidual1, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channel, hid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=3, stride=2, padding=1, groups=hid_channel, bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, hid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = torch.cat((self.conv1(x), self.conv2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class shffulnetv2(nn.Module):
    def __init__(self, num_classes=3):
        super(shffulnetv2, self).__init__()
        block0=InvertedResidual0
        block1=InvertedResidual1
        layers=[]
        layers.extend([
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            block1(24,58),
            block0(116),
            block0(116),
            block0(116),
            block1(116,116),
            block0(232),
            block0(232),
            block0(232),
            block0(232),
            block0(232),
            block0(232),
            block0(232),
            block1(232, 232),
            block0(464),
            block0(464),
            block0(464),
            nn.Conv2d(464, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
                       ])
        self.conv=nn.Sequential(*layers)
        self.fc = nn.Linear(1024, num_classes)

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
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

