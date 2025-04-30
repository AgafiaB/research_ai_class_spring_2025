import torch.nn as nn
class Inv2d(nn.Module):
    def __init__(self, channels, reduction=1, kernel_size=3, group_ch=1, stride=1, padding=1, dilation=1):
        super().__init__()

        self.k = kernel_size
        self.group_ch = group_ch
        self.in_channels = channels
        self.out_channels = channels
        self.g = channels // group_ch
        self.r = reduction

        self.stride=stride
        self.padding = padding
        self.dilation=dilation

        # avg pool is for adjusting the kernel and therefore output based on the stride
        self.o = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
        self.reduce = nn.Conv2d(channels, channels // reduction, 1)
        self.batch_norm = nn.BatchNorm2d(channels//reduction)
        self.span = nn.Conv2d(channels // reduction, self.g*(kernel_size**2), 1)
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride, dilation=dilation)
    
    def forward(self, X):
        '''
        Input should be an image tensor of shape: (batch_size, channels, h, w)
        '''
        W = self.reduce(self.o(X))
        # print('W.shape', W.shape)
        W = nn.ReLU()(self.batch_norm(W))
        W = self.span(W)
        # print('W.shape', W.shape)
        b, c, h, w = W.shape
        W = W.view(b, self.g, 1, self.k**2, w*h)
        print(W.shape)

        patches = self.unfold(X)
        patches = patches.view(b, self.g, self.group_ch, self.k**2, w*h)
        out = patches*W # (b, g, g//c, k*k, w*h)

        # output width and height should be w / stride and h / stride

        out = out.view(b, self.out_channels, self.k**2, w*h).sum(dim=2)
        out = out.view(b, self.out_channels, w, h)

        return out