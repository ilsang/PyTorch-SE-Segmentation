
import torch
import torch.nn as nn
import torch.nn.functional as F

################ Unet  ################

class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(UNetDec, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            self.excitation1 = nn.Conv2d(out_channels, out_channels // self.reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv2d(out_channels // self.reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.up(x)
        if self.reduction_ratio:
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out,fm_size)
            scale_weight = F.relu(self.excitation1(scale_weight))
            scale_weight = F.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out

class UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio, dropout=False):
        super(UNetEnc, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
                       
        ]
        if dropout:
            layers += [nn.Dropout(.5)]

        self.down = nn.Sequential(*layers)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            self.excitation1 = nn.Conv2d(out_channels, out_channels // self.reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv2d(out_channels // self.reduction_ratio, out_channels, kernel_size=1)

        
    def forward(self, x):
        out = self.down(x)
        if self.reduction_ratio:
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out,fm_size)
            scale_weight = F.relu(self.excitation1(scale_weight))
            scale_weight = F.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out
    
class Bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, reduction_ratio, dropout=False):
        super(Bottleneck_block, self).__init__()

        layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]

        if dropout:
            layers += [nn.Dropout(.5)]
            
        self.center = nn.Sequential(*layers)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            self.excitation1 = nn.Conv2d(out_channels, out_channels // self.reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv2d(out_channels // self.reduction_ratio, out_channels, kernel_size=1)

        
    def forward(self, x):
        out = self.center(x)
        if self.reduction_ratio:
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out,fm_size)
            scale_weight = F.relu(self.excitation1(scale_weight))
            scale_weight = F.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out

class UNet(nn.Module):

    def __init__(self, num_classes, init_features, network_depth, bottleneck_layers, reduction_ratio):
        super(UNet, self).__init__()
        
        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.bottleneck_layers = bottleneck_layers
        skip_connection_channel_counts = []

        self.add_module('firstconv', nn.Conv2d(in_channels=num_classes, 
                  out_channels=init_features, kernel_size=3, 
                  stride=1, padding=1, bias=True))
        
        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            self.encodingBlocks.append(UNetEnc(features, 2*features, reduction_ratio=self.reduction_ratio))
            skip_connection_channel_counts.insert(0, 2*features)
            features *= 2
        final_encoding_channels = skip_connection_channel_counts[0]
        
        self.bottleNecks = nn.ModuleList([])
        for i in range(self.bottleneck_layers):
            dilation_factor = 1
            self.bottleNecks.append(Bottleneck_block(final_encoding_channels,
                                              final_encoding_channels, dilation_rate=dilation_factor,
                                                    reduction_ratio=self.reduction_ratio))
        
        self.decodingBlocks = nn.ModuleList([])
        for i in range(self.network_depth):
            if i == 0:
                prev_deconv_channels = final_encoding_channels
            self.decodingBlocks.append(UNetDec(prev_deconv_channels + skip_connection_channel_counts[i],
                                              skip_connection_channel_counts[i],
                                               reduction_ratio=self.reduction_ratio))
            prev_deconv_channels = skip_connection_channel_counts[i]
        
        self.final = nn.Conv2d(2 * init_features, num_classes, 1)
        
    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(self.network_depth):
            out = self.encodingBlocks[i](out)
            skip_connections.append(out)
            
        for i in range(self.bottleneck_layers):
            out = self.bottleNecks[i](out)
            
        for i in range(self.network_depth):
            skip = skip_connections.pop()
            out = self.decodingBlocks[i](torch.cat([out, skip], 1))

        out = self.final(out)
        return out

# class UNet(nn.Module):

#     def __init__(self, num_classes, init_features, network_depth, bottleneck_layers, reduction_ratio):
#         super(UNet, self).__init__()
        
#         self.reduction_ratio = reduction_ratio
#         self.network_depth = network_depth
#         self.bottleneck_layers = bottleneck_layers
#         skip_connection_channel_counts = []
        
#         self.add_module('firstconv', nn.Conv2d(in_channels=num_classes, 
#                   out_channels=init_features, kernel_size=3, 
#                   stride=1, padding=1, bias=True))
        
        
#         self.encodingBlocks = nn.ModuleList([])
#         features = init_features
        
#         for i in range(self.network_depth):
#             self.encodingBlocks.append(UNetEnc(features, 2*features))
            
#             skip_connection_channel_counts.insert(0, 2*features)
#             features *= 2
#         final_encoding_channels = skip_connection_channel_counts[0]
        
        
#         self.bottleNecks = nn.ModuleList([])
#         for i in range(self.bottleneck_layers):
#             dilation_factor = 1
#             self.bottleNecks.append(Bottleneck_block(final_encoding_channels,
#                                               final_encoding_channels, dilation_rate=dilation_factor))
            
        
#         self.decodingBlocks = nn.ModuleList([])
#         for i in range(self.network_depth):
#             if i == 0:
#                 prev_deconv_channels = final_encoding_channels
#             self.decodingBlocks.append(UNetDec(prev_deconv_channels + skip_connection_channel_counts[i],
#                                               skip_connection_channel_counts[i], self.reduction_ratio))
#             prev_deconv_channels = skip_connection_channel_counts[i]
        
        
#         self.final = nn.Conv2d(2 * init_features, num_classes, 1)

        
#     def forward(self, x):
        
#         out = self.firstconv(x)
        
#         skip_connections = []
#         for i in range(self.network_depth):
#             out = self.encodingBlocks[i](out)
#             skip_connections.append(out)
            
#         for i in range(self.bottleneck_layers):
#             out = self.bottleNecks[i](out)
            
#         for i in range(self.network_depth):
#             skip = skip_connections.pop()
#             out = self.decodingBlocks[i](torch.cat([out, skip], 1))
            
#         out = self.final(out)
#         return out
    
    

################ Densenet  ################   
    
def center_crop(layer, max_height, max_width):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    #Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        
        #author's impl - lasange 'same' pads with half 
        # filter size (rounded down) on "both" sides
        self.add_module('conv', nn.Conv2d(in_channels=in_channels, 
                out_channels=growth_rate, kernel_size=3, stride=1, 
                  padding=1, bias=True))
        
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        out = super(DenseLayer, self).forward(x)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, reduction_ratio, upsample=False):
        super(DenseBlock, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])
        
        if self.reduction_ratio is not None:
            if self.upsample:
                self.excitation1 = nn.Conv2d(growth_rate * n_layers,
                                                      growth_rate * n_layers // self.reduction_ratio, kernel_size=1)
                self.excitation2 = nn.Conv2d(growth_rate * n_layers // self.reduction_ratio, 
                                                      growth_rate * n_layers, kernel_size=1)
            else:
                self.excitation1 = nn.Conv2d((in_channels+growth_rate * n_layers),
                                             (in_channels+growth_rate * n_layers) // self.reduction_ratio,
                                             kernel_size=1)
                self.excitation2 = nn.Conv2d((in_channels+growth_rate * n_layers) // self.reduction_ratio, 
                                             (in_channels+growth_rate * n_layers), kernel_size=1)

        
    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            out = torch.cat(new_features,1)
            
            if self.reduction_ratio is not None:
                fm_size = out.size()[2]
                scale_weight = F.avg_pool2d(out, fm_size)
                scale_weight = F.relu(self.excitation1(scale_weight))
                scale_weight = F.sigmoid(self.excitation2(scale_weight))
                out = out * scale_weight.expand_as(out)
            return out
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
                
            if self.reduction_ratio is not None:
                fm_size = x.size()[2]
                scale_weight = F.avg_pool2d(x, fm_size)
                scale_weight = F.relu(self.excitation1(scale_weight))
                scale_weight = F.sigmoid(self.excitation2(scale_weight))
                x = x * scale_weight.expand_as(x)
            return x
    
class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels=in_channels, 
              out_channels=in_channels, kernel_size=1, stride=1, 
                padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))
        
    def forward(self, x):
        out = super(TransitionDown, self).forward(x)
        return out
    
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, 
               out_channels=out_channels, kernel_size=3, stride=2, 
              padding=0, bias=True) #crop = 'valid' means padding=0. Padding has reverse effect for transpose conv (reduces output size)
        #http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html#lasagne.layers.TransposedConv2DLayer
        #self.updample2d = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out
    
class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers, reduction_ratio):
        self.reduction_ratio = reduction_ratio
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate, n_layers,
                                                 self.reduction_ratio, upsample=True))

    def forward(self, x):
        out = super(Bottleneck, self).forward(x)
        return out

    
# def FCDenseNet57(n_classes):
#     return FCDenseNet(in_channels=2, down_blocks=(4, 4, 4, 4, 4), 
#                  up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4, 
#                  growth_rate=4, out_chans_first_conv=48, n_classes=n_classes)

# def FCDenseNet67(n_classes):
#     return FCDenseNet(in_channels=2, down_blocks=(5, 5, 5, 5, 5), 
#                  up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, 
#                  growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)

# def FCDenseNet103(n_classes):
#     return FCDenseNet(in_channels=2, down_blocks=(4,5,7,10,12), 
#                  up_blocks=(12,10,7,5,4), bottleneck_layers=15, 
#                  growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)  

class FCDenseNet(nn.Module):
    def __init__(self, reduction_ratio=None, in_channels=2, down_blocks=(4,5,7,10,12), 
                 up_blocks=(12,10,7,5,4), bottleneck_layers=15, 
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.reduction_ratio = reduction_ratio

        # First Convolution #
        
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_chans_first_conv, kernel_size=3, 
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv
        skip_connection_channel_counts = []
        
        # Downsampling path #
        
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i], self.reduction_ratio))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
            
        #     Bottleneck    #
        
        self.add_module('bottleneck',Bottleneck(cur_channels_count, 
                                     growth_rate, bottleneck_layers, self.reduction_ratio))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels 
        
        #   Upsampling path   #
        
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i], self.reduction_ratio, upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        #One final dense block
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1], self.reduction_ratio,
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        #      Softmax      #
        
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, 
               out_channels=n_classes, kernel_size=1, stride=1, 
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        out = self.firstconv(x)
        
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            
        out = self.bottleneck(out)
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
            
        out = self.finalConv(out)
        out = self.softmax(out)
        return out
    