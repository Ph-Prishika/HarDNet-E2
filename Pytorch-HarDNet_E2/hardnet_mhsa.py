import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace=True))


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, stride=1, bias=False):
        super().__init__()
        self.add_module('dwconv', nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                            stride=stride, padding=1, groups=in_channels, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(in_channels))


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, stride=stride))


class MHSA(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
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
        out_channels = int((out_channels + 1) // 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers,
                 keepBase=False, residual_out=False, dwconv=False,
                 add_mhsa=False, heads=4):
        super().__init__()
        self.keepBase = keepBase
        self.residual_out = residual_out
        self.add_mhsa = add_mhsa
        self.links = []
        self.out_channels = 0

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            out_ch, in_ch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            conv_layer = CombConvLayer(in_ch, out_ch) if dwconv else ConvLayer(in_ch, out_ch)
            self.layers.append(conv_layer)
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += out_ch

        if self.add_mhsa:
            self.mhsa = MHSA(self.out_channels, heads=heads)

    def forward(self, x):
        layers_ = [x]
        for layer_idx, layer in enumerate(self.layers):
            inputs = [layers_[i] for i in self.links[layer_idx]]
            x_in = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            out = layer(x_in)
            layers_.append(out)

        outputs = []
        for i in range(len(layers_)):
            if (i == 0 and self.keepBase) or (i == len(layers_) - 1) or (i % 2 == 1):
                outputs.append(layers_[i])
        out = torch.cat(outputs, dim=1)

        if self.add_mhsa:
            out = self.mhsa(out)

        return out
class HarDNet(nn.Module):
    def __init__(self, depth_wise=False, arch=85, pretrained=False, weight_path=''):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.5
        drop_rate = 0.1
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            #gr = [20, 20, 24, 32, 44, 224]  # reduced growth rate
            gr = [16, 16, 20, 28, 40, 192]  # reduced growth rate
            #gr       = [  24,  24,  28,  36,  48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.4
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        self.base = nn.ModuleList()
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == len(n_layers) - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2) if max_pool else DWConvLayer(ch, ch, stride=2))

        self.base.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(ch, 1000)))

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        return x
