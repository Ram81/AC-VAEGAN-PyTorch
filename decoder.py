import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()

        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2,
                                       stride=2, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Decoder(nn.Module):
    def __init__(self, z_size, size, num_classes=10):
        super(Decoder, self).__init__()

        # start from B * z_size
        # concatenate one hot encoded class vector
        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + num_classes), out_features=(8 * 8 * size), bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                nn.ReLU(True))
        self.size = size

        layers = [
            DecoderBlock(channel_in=self.size, channel_out=self.size),
            DecoderBlock(channel_in=self.size, channel_out=self.size // 2)
        ]

        self.size = self.size // 2
        layers.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 4))
        self.size = self.size // 4

        # final conv to get 3 channels and tanh layer
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))
        self.conv = nn.Sequential(*layers)

    def forward(self, ten, one_hot_classes):
        ten_cat = torch.cat((one_hot_classes, ten), 1)
        ten = self.fc(ten_cat)
        ten = ten.view(len(ten), -1, 8, 8)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)
