"""model_simple module

Note:
    1. supports 32x32 image size only.
    2. not supports latent feature skip-connections.
    3. not supports vanishing residuals.
    4. not supports layer reptitions.
"""

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, image_size, hidden_dim=64, n_filter=128, n_repeat=2):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_filter = n_filter
        self.n_repeat = n_repeat

        self.fc = nn.Linear(self.hidden_dim, 8*8*self.n_filter)
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
            nn.ELU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
            nn.ELU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.n_filter, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, h):
        h0 = self.fc(h)
        h0 = h0.view(h0.size(0), self.n_filter, 8, 8)
        out = self.conv(h0)

        return out


class Encoder(nn.Module):
    def __init__(self, image_size, hidden_dim=64, n_filter=128, n_repeat=2):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_filter = n_filter
        self.n_repeat = n_repeat

        self.conv = nn.Sequential(
           nn.Conv2d(3, self.n_filter, 3, 1, 1),
           nn.ELU(True),
           nn.Conv2d(self.n_filter, self.n_filter, 3, 1, 1),
           nn.ELU(True),
           nn.Conv2d(self.n_filter, self.n_filter*2, 3, 2, 1),
           nn.ELU(True),
           nn.Conv2d(self.n_filter*2, self.n_filter*2, 3, 1, 1),
           nn.ELU(True),
           nn.Conv2d(self.n_filter*2, self.n_filter*3, 3, 2, 1),
           nn.ELU(True),
           nn.Conv2d(self.n_filter*3, self.n_filter*3, 3, 1, 1),
           nn.ELU(True),
           nn.Conv2d(self.n_filter*3, self.n_filter*3, 3, 1, 1),
           nn.ELU(True),
        )
        self.fc = nn.Linear(8*8*self.n_filter*3, self.hidden_dim)

    def forward(self, image):
        out = self.conv(image)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
