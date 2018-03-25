"""model_skip_repeat module

Note:
    1. supports any image sizes that meet (power of 2) and (bigger than or equal to 32)
    2. supports latent feature skip-connections.
    3. supports layer reptitions.
    4. not supports vanishing residuals.
"""

from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F


def base_decoder_block(_type, n_filter=128, n_repeat=2):
    layers = []
    if _type == 'front':
        for i in range(n_repeat):
            layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))

    elif _type == 'inter':
        for i in range(n_repeat):
            if i == 0:
                layers.append(nn.Conv2d(2*n_filter, n_filter, 3, 1, 1))
            else:
                layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))

    elif _type == 'end':
        for i in range(n_repeat):
            if i != (n_repeat-1):
                layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))
            else:
                layers.append(nn.Conv2d(n_filter, 3, 3, 1, 1))
                layers.append(nn.Tanh())

    else:
        raise

    return layers


def base_encoder_block(_type, n_filter=128, n_repeat=2, inter_scale=1):
    m = inter_scale

    layers = []
    if _type == 'front':
        for i in range(n_repeat):
            if i == 0:
                layers.append(nn.Conv2d(3, n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))
            else:
                layers.append(nn.Conv2d(n_filter, n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))

    elif _type == 'inter':
        for i in range(n_repeat):
            layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            if i != (n_repeat-1):
                layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
                layers.append(nn.ELU(True))
            else:
                layers.append(nn.Conv2d(m*n_filter, (m+1)*n_filter, 3, 2, 1))
                layers.append(nn.ELU(True))

    elif _type == 'end':
        for i in range(n_repeat):
            layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(m*n_filter, m*n_filter, 3, 1, 1))
            layers.append(nn.ELU(True))

    else:
        raise

    return layers


class Decoder(nn.Module):
    def __init__(self, image_size, hidden_dim=64, n_filter=128, n_repeat=2):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.n_upsample = int(log2(image_size//8))
        self.hidden_dim = hidden_dim
        self.n_filter = n_filter
        self.n_repeat = n_repeat

        self.fc = nn.Linear(self.hidden_dim, 8*8*self.n_filter)
        self.convs = dict()
        for i in range(self.n_upsample+2):
            if i == 0:
                self.convs[i] = nn.Sequential(*base_decoder_block('front', n_filter, n_repeat))
                self.add_module(name='front', module=self.convs[i])
            elif i <= self.n_upsample:
                self.convs[i] = nn.Sequential(*base_decoder_block('inter', n_filter, n_repeat))
                self.add_module(name='inter'+str(i), module=self.convs[i])
            else:
                self.convs[i] = nn.Sequential(*base_decoder_block('end', n_filter, n_repeat))
                self.add_module(name='end', module=self.convs[i])


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, h):
        h0 = self.fc(h)
        h0 = h0.view(h0.size(0), self.n_filter, 8, 8)

        out = self.convs[0](h0)
        out = torch.cat([out, h0], dim=1)
        out = F.upsample(out, scale_factor=2, mode='nearest')

        for i in range(1, self.n_upsample):
            out = self.convs[i](out)
            h0 = F.upsample(h0, scale_factor=2, mode='nearest')
            out = torch.cat([out, h0], dim=1)
            out = F.upsample(out, scale_factor=2, mode='nearest')

        out = self.convs[i+1](out)
        out = self.convs[i+2](out)

        return out


class Encoder(nn.Module):
    def __init__(self, image_size, hidden_dim=64, n_filter=128, n_repeat=2):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.n_upsample = int(log2(self.image_size//8))
        self.hidden_dim = hidden_dim
        self.n_filter = n_filter
        self.n_repeat = n_repeat

        self.convs = dict()
        for i in range(self.n_upsample+2):
            if i == 0:
                self.convs[i] = nn.Sequential(*base_encoder_block('front', self.n_filter, self.n_repeat))
                self.add_module('front', self.convs[i])
            elif i <= self.n_upsample:
                self.convs[i] = nn.Sequential(*base_encoder_block('inter', self.n_filter, self.n_repeat, i))
                self.add_module('inter'+str(i), self.convs[i])
            else:
                self.convs[i] = nn.Sequential(*base_encoder_block('end', self.n_filter, self.n_repeat, i))
                self.add_module('end', self.convs[i])

        self.fc = nn.Linear(8*8*i*self.n_filter, self.hidden_dim)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, image):
        out = self.convs[0](image)
        for i in range(1, len(self.convs.keys())):
            out = self.convs[i](out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
