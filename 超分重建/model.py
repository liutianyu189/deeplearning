import torch
import torch.nn as nn

class fsrcnn(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(fsrcnn, self).__init__()
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
                                        nn.PReLU())
        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(4):
            self.layers.append(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)
        self.last_part = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=9, stride=upscale_factor, padding=3, output_padding=1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()