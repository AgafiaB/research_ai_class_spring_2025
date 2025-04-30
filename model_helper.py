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
    

from torchvision.ops import Conv2dNormActivation

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, norm_layer=None):
        super().__init__()

        assert stride == 1 or stride == 2, 'stride must be 1 or 2'

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(round(in_channels * expand_ratio))
        # print(hidden_dim)

        layers = [] 

        if expand_ratio != 1:
            # add expansion 
            layers.append(Conv2dNormActivation(in_channels, hidden_dim, kernel_size=3, stride=stride, 
                                               norm_layer=norm_layer, activation_layer=nn.ReLU6))
        # depthwise conv => pointwise => norm_layer
        # hidden_dim == in_channels if expand_ratio==1
        # b/c groups = hidden_dim, each input channel is convolved with its own set of filters
            # b/c instead of using multiple 3d kernels, we want multiple 2d kernels that each go across a different channel
        layers.extend([Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation_layer=nn.ReLU6), 
                                                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False), 
                                                norm_layer(out_channels)])
        
        self.conv = nn.Sequential(*layers)
        
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self._is_cn = stride > 1 # downsample indicator - what does this mean?
        self.use_res_conn = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_conn:
            return x + self.conv(x)
        else:                
            return self.conv(x)
        

def _make_divisible(v, divisor, min_value=None): 
    '''
    Description: rounds v to nearest multiple of divisor
    '''
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

import torch 

class MobileNetV2(nn.Module):
    """
        MobileNet V2 main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
    def __init__(self, width_mult=1.0, inverted_residual_setting = None, round_nearest=8,
                  block = None, norm_layer = None, dropout = 0.2):
        super().__init__()
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        input_channel = 32 # adjustable
        last_channel = 1280 # adjustable

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t - expansion factor, c - # of output after block, n - # of repitions, s - stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # build the first layer
        # input_channel is the number of channels after the first Conv2dNormActivation
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        
        features = [
            Conv2dNormActivation(3, input_channel, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]

        # build the inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel 
                
        # build last severals
        features.append(Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))

        # complete CNN architecture
        self.features = nn.Sequential(*features) 

        # weight initialization
        for m in self.modules():
            # print('initializing weights')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def _forward_impl(self, x):
        x = self.features(x) # pass input through layers
        
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)) # takes the average value for each each channel
        x = torch.flatten(x, 1) # flatten for linear layer
        # x = self.classifier(x) # classify
        return x

    def forward(self, x):
        return self._forward_impl(x)

        