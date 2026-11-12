import torch
import torch.nn as nn

class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, h, w = x.size()

        device = x.device
        dtype = x.dtype

        xx = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        yy = torch.linspace(-1, 1, h, device=device, dtype=dtype)

        xx = xx.view(1, 1, 1, w).expand(b, -1, h, -1)
        yy = yy.view(1, 1, h, 1).expand(b, -1, -1, w)

        x = torch.cat([x, xx, yy], dim=1)

        return x


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False):
        super().__init__()

        self.addcoords = AddCoords()

        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        x = self.addcoords(x)
        return self.conv(x)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation

    @property
    def groups(self):
        return self.conv.groups
