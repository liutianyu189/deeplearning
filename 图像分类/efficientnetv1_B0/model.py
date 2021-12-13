import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
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

class InvertedResidual0(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InvertedResidual0, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Hardswish(),
            SqueezeExcitation(in_channel,in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Identity()
        )
    def forward(self, x):
            return self.conv(x)

class InvertedResidual1(nn.Module):
    def __init__(self, in_channel, out_channel,drop_num=1.0,expand=6,strid=1,pd=1,ks=3):
        super(InvertedResidual1, self).__init__()
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
            nn.Conv2d(hiden_channel, hiden_channel, kernel_size=ks, stride=strid, padding=pd, groups=in_channel, bias=False),
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

class EfficientNet(nn.Module):
    def __init__(self,num_classes=3):
        super(EfficientNet, self).__init__()
        block0 = InvertedResidual0
        block1 = InvertedResidual1
        layers=[]
        layers.extend([
            nn.Conv2d(3, 32, kernel_size=3, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        ])
        layers.append(block0(32,16))
        layers.extend([
            block1(16,24,strid=2),
            block1(24,24,drop_num=0.1),
        ])
        layers.extend([
            block1(24, 40, strid=2,ks=5,pd=2),
            block1(40, 40,drop_num=0.2,ks=5,pd=2),
        ])
        layers.extend([
            block1(40, 80, strid=2),
            block1(80, 80,drop_num=0.3),
            block1(80, 80,drop_num=0.4)
        ])
        layers.extend([
            block1(80, 112,ks=5,pd=2),
            block1(112, 112,drop_num=0.5,ks=5,pd=2),
            block1(112, 112, drop_num=0.5,ks=5, pd=2)
        ])
        layers.extend([
            block1(112, 192,ks=5,pd=2,strid=2),
            block1(192, 192,drop_num=0.7,ks=5,pd=2),
            block1(192, 192, drop_num=0.8,ks=5, pd=2),
            block1(192, 192, drop_num=0.9,ks=5, pd=2)
        ])
        layers.extend([
            block1(192, 320)
        ])
        layers.extend([
            nn.Conv2d(320,1280,kernel_size=1,bias=False),
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

