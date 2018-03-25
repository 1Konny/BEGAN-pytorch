"""model.py"""

import torch.nn as nn

import models.model_simple as simple
import models.model_skip as skip
import models.model_skip_repeat as skip_repeat


def encoder(_type, image_size, hidden_dim, n_filter, n_repeat):
    if _type == 'simple':
        return simple.Encoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip':
        return skip.Encoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip_repeat':
        return skip_repeat.Encoder(image_size, hidden_dim, n_filter, n_repeat)
    else:
        return None


def decoder(_type, image_size, hidden_dim, n_filter, n_repeat):
    if _type == 'simple':
        return simple.Decoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip':
        return skip.Decoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip_repeat':
        return skip_repeat.Decoder(image_size, hidden_dim, n_filter, n_repeat)
    else:
        return None


class Discriminator(nn.Module):
    def __init__(self, _type, image_size, hidden_dim, n_filter, n_repeat):
        super(Discriminator, self).__init__()
        self.encode = encoder(_type, image_size, hidden_dim, n_filter, n_repeat)
        self.decode = decoder(_type, image_size, hidden_dim, n_filter, n_repeat)

    def weight_init(self, mean, std):
        self.encode.weight_init(mean, std)
        self.decode.weight_init(mean, std)

    def forward(self, image):
        out = self.encode(image)
        out = self.decode(out)

        return out


class Generator(nn.Module):
    def __init__(self, _type, image_size, hidden_dim, n_filter, n_repeat):
        super(Generator, self).__init__()
        self.decode = decoder(_type, image_size, hidden_dim, n_filter, n_repeat)

    def weight_init(self, mean, std):
        self.decode.weight_init(mean, std)

    def forward(self, h):
        out = self.decode(h)

        return out
