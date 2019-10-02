import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import EncoderBlock


class Discriminator(nn.Module):
    def __init__(self, channels_in=3, recon_level=3, num_classes=10):
        super(Discriminator, self).__init__()
        self.size = channels_in
        self.recon_level = recon_level

        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        ))

        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))

        # final fc layer to get the score (real or fake)

        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True)
        )
        self.fc_disc = nn.Linear(in_features=512, out_features=1)
        self.fc_aux = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, ten, other_ten, mode='REC'):
        ten = torch.cat((ten, other_ten), 0)
        if mode == 'REC':
            for i, layer in enumerate(self.conv):
                # take 9th layer as one of the outputs
                if i == self.recon_level:
                    ten, layer_ten = layer(ten, True)
                    # fetch the layer representations just for the original & reconstructed,
                    # flatten, because it is multidimensional
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = layer(ten)
        else:
            for i, layer in enumerate(self.conv):
                ten = layer(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            ten_disc = self.fc_disc(ten)
            ten_aux = self.fc_aux(ten)
            return F.sigmoid(ten_disc), F.log_softmax(ten_aux)

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)

