import torch.nn as nn
from model_helper import Inv2d


# residual block
class Bottleneck(nn.Module):
    '''
    in_channels: input channels of this block
    out_channels: output channels of this block
    expansion: the ration of out_channels/mid_channels
        where mid_channels is the input/output channels of conv2
        default: 4
    stride: default: 1
    dilation: default: 1
    '''

    def __init__(self, in_channels, out_channels, 
                 expansion=4, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation

        self.conv1_stride = 1
        self.conv2_stride = stride # for involution 

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=self.mid_channels, 
                               kernel_size=1, 
                               stride=self.conv1_stride, 
                               bias=False)
        
        self.conv2 = Inv2d(self.mid_channels,kernel_size=3, stride=self.conv2_stride, reduction=4, group_ch=self.mid_channels // 2)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, 
                               kernel_size=1, bias=False)
        
        self.norm1 = nn.BatchNorm2d(self.mid_channels)
        self.norm2 = nn.BatchNorm2d(self.mid_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def _inner_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.stride != 1 or self.in_channels != self.out_channels:
            downsample = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                   stride=self.stride, kernel_size=1, bias=False)
            identity = downsample(identity)
        
        out = identity + out
        return out

    def forward(self, x):
        out = self._inner_forward(x)

        out = self.relu(out)

        return out
    

# TODO: Take out the dilation parameter to simplify since we will just be using ones anyway
class ResLayer(nn.Sequential):
    '''
    block: residual block used to build ResLayer -- typically of the Bottleneck class
    num_blocks: number of blocks
    in_channels: input channels of this block
    out_channels: output channels of this block
    expansion: expansion for Bottleneck -- default is 4
    stride: stride of the first block -- default is 1
    '''

    def __init__(self, block, num_blocks, in_channels, out_channels, 
                 expansion=4, stride=1):
        self.block = block
        self.expansion = expansion

        layers = []
        layers.append(
            block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                expansion=self.expansion, 
                stride=stride, 
            )
        )

        in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(
                block(in_channels=in_channels, 
                      out_channels=out_channels, 
                      expansion=self.expansion, 
                      stride=1)
            )
        
        # b/c we are inheriting from nn.Sequential
        super(ResLayer, self).__init__(*layers) 
    

class RedNet(nn.Module):
    '''
    Parameters:
        depth: network depth from {26, 38, 50, 101, 152}
        in_channels: number of input image channels. default: 3
        stem_channels: output channels of the stem/beginning layer. default: 64
        base_channels: middle channels of the first stage. default: 64
        num_stages: stages of the network. default: 4
        strides: strides of the first block of each stage. default: (1, 2, 2, 2)
        dilations: dilations of each stage. default: (1, 1, 1, 1)
        expansion: default: 4
    '''

    # settings based on the input depth: 
    arch_settings = {
        26: (Bottleneck, (1, 2, 4, 1)),
        38: (Bottleneck, (2, 3, 5, 2)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, in_channels=3, stem_channels=64, base_channels=64, 
                 num_stages=4, strides=(1, 1, 1, 1), dilations=(1, 1, 1, 1), 
                 expansion=4):
        super(RedNet, self).__init__()

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.expansion = expansion
        assert num_stages >= 1 and num_stages <= 4

        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages

        self.block, stage_blocks = self.arch_settings[depth] # self.block is a Bottleneck object

        self._make_stem_layer(in_channels=in_channels, stem_channels=stem_channels)

        self.res_layers = []

        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self._make_res_layer(self.block, num_blocks, _in_channels,
                                             _out_channels, self.expansion, stride 
                                             )
            _in_channels = _out_channels
            _out_channels = _out_channels*2

            layer_name = f'layer{i+1}' # COMMENT: this is super cool!
            self.add_module(layer_name, res_layer) # ~ inheritance ~
            self.res_layers.append(layer_name)
        
        self.feat_dim = res_layer[-1].out_channels # COMMENT: what is the purpose of this?


    def _make_stem_layer(self, in_channels, stem_channels):
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels // 2, 
                      kernel_size=3, stride=2, padding=1), 
            Inv2d(in_channels=stem_channels // 2, kernel_size=3, stride=1, group_ch=in_channels // 2), 
            nn.BatchNorm2d(stem_channels // 2), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, 
                      stride=1, padding=1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # same conv

    def init_weights(self, pretrained=None):
        super(RedNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m, 1) # initialize with ones
    
    def _make_res_layer(self, block, num_blocks, in_channels, out_channels, 
                        expansion, stride):
        return ResLayer(block, num_blocks, in_channels, out_channels, expansion, stride)
                
    
    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name) 
            x = res_layer(x)
        
        return x
            
