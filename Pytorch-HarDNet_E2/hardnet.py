import torch
import torch.nn as nn

# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Basic conv block
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

# Squeeze-and-Excitation block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Modified HarDBlock with SE and Residual
class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul, skip_layers):
        if layer == 0:
            return base_ch, 0, []

        out_channels = growth_rate
        link = []

        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul

        # Extra skip connections
        for skip in skip_layers:
            if skip < layer:
                link.append(skip)

        link = sorted(set(link))
        out_channels = int(int(out_channels + 1) / 2) * 2

        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul, skip_layers)
            in_channels += ch

        return out_channels, in_channels, link

    def __init__(self, in_channels, growth_rate, grmul, n_layers,
                 keepBase=False, residual_out=True, skip_layers=[2, 4, 6]):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        self.layers_ = nn.ModuleList()
        self.out_channels = 0
        self.skip_layers = skip_layers
        self.residual_out = residual_out

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul, skip_layers)
            self.links.append(link)
            self.layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.out_indexes = [0] if keepBase else []
        self.out_indexes += [i + 1 for i in range(n_layers) if (i == n_layers - 1 or i % 2 == 0)]

        self.se = SELayer(self.out_channels)

        if self.residual_out:
            if in_channels != self.out_channels:
                self.residual_layer = ConvLayer(in_channels, self.out_channels, kernel=1)
            else:
                self.residual_layer = nn.Identity()
        else:
            self.residual_layer = None

    def forward(self, x):
        layers_ = [x]
        for i in range(len(self.layers_)):
            link = self.links[i]
            inputs = [layers_[j] for j in link] if link else [x]
            x_in = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            out = self.layers_[i](x_in)
            layers_.append(out)

        out = torch.cat([layers_[i] for i in self.out_indexes], dim=1)
        out = self.se(out)

        if self.residual_out:
            res = self.residual_layer(layers_[0])
            out = out + res

        return out

# HarDNet backbone with SE and extra skip
class HarDNet(nn.Module):
    def __init__(self, arch=85, pretrained=False, weight_path='', num_classes=1000):
        super().__init__()
        first_ch = [48, 96]
        ch_list = [192, 256, 320, 480, 720, 1280]
        gr = [ 24,  24,  28,  36,  48, 256]
        n_layers = [8, 16, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 0, 1, 0]
        grmul = 1.7
        drop_rate = 0.2

        self.base = nn.ModuleList()
        self.base.append(ConvLayer(3, first_ch[0], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], skip_layers=[2, 4, 6])
            ch = blk.out_channels
            self.base.append(blk)
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.base.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(ch, num_classes)
        ))

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        return x
#BERT/last_code_may/Pytorch-HarDNet_new_SE_resi/hardnet.py