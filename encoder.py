import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()

        # convolution layer to halve the dimensions

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        ten = self.conv(ten)
        ten_out = ten
        ten = self.bn(ten)
        ten = F.relu(ten, False)

        # if out=True return intermediate output for reconstruction error
        if out:
            return ten, ten_out
        return ten


class Encoder(nn.Module):
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers = []

        # the first time 3->64, for every other double the channel size
        for i in range(3):
            if i == 0:
                layers.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape B x 256 x 8 x 8
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True))
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)
