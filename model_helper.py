import torch.nn as nn
class Inv2d(nn.Module):
    def __init__(self, channels, reduction=1, kernel_size=3, group_ch=1, stride=1, padding=0, dilation=1):
        self.k = kernel_size
        self.group_ch = group_ch
        self.in_channels = channels
        self.out_channels = channels
        self.g = channels // group_ch
        self.r = reduction

        self.stride=stride
        self.padding = padding
        self.dilation=dilation

        self.o = nn.AvgPool1d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
        self.reduce = nn.Conv2d(channels, channels // reduction, 1)
        self.span = nn.Conv2d(channels // reduction, self.g*(kernel_size**2), 1)
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride, dilation=dilation)
    
    def forward(self, X):
        W = self.span(self.reduce(self.o(X)))
        b, c, h, w = W.shape
        W = W.view(b, self.g, 1, self.k**2, w*h)

        patches = self.unfold(X)
        patches = patches.view(b, self.g, self.group_ch, self.k**2, w*h)
        out = patches*W # (b, g, g//c, k*k, w*h)

        # out shape should be same as in shape

        out = out.view(b, self.out_channels, self.k**2, w*h).sum(dim=2)
        out = out.view(b, self.out_channels, w, h)

        return out