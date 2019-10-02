import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from torch.autograd import Variable


class VAEGAN(nn.Module):
    def __init__(self, z_size=128, recon_level=3):
        super(VAEGAN, self).__init__()

        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)

        self.discriminator = Discriminator(channels_in=3, recon_level=recon_level)

        # initialize self defined params
        training = True
        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten, gen_size=10):
        if self.training:
            # save original images
            ten_original = ten

            # encode
            mu, log_variances = self.encoder(ten)

            # we need true variance not log
            variances = torch.exp(log_variances * 0.5)

            # sample from gaussian
            ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)

            # shift and scale using mean and variances
            ten = ten_from_normal * variances + mu

            # decode tensor
            ten = self.decoder(ten)

            # discriminator for reconstruction
            ten_layer = self.discriminator(ten, ten_original, mode='REC')

            # decode from samples
            ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)

            ten = self.decoder(ten_from_normal)
            ten_real_fake, ten_aux = self.discriminator(ten_original, ten, mode='GAN')

            return ten, ten_real_fake, ten_layer, mu, log_variances, ten_aux
        else:
            if ten is None:
                # just sample and decode
                ten = Variable(torch.randn(gen_size, self.z_size).cuda(), requires_grad=False)
            else:
                mu, log_variances = self.encoder(ten)
                # we need true variance not log
                variances = torch.exp(log_variances * 0.5)

                # sample from gaussian
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)

                # shift and scale using mean and variances
                ten = ten_from_normal * variances + mu

            # decode tensor
            ten = self.decoder(ten)
            return ten

    def __call__(self, *args, **kwargs):
        return super(VAEGAN, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original, labels_sampled,
             mu, variances, aux_labels_predicted, aux_labels_sampled, aux_labels_original):
        """
        :param ten_original: original images
        :param ten_predict: predicted images (decode ouput)
        :param layer_original: intermediate layer for original (intermediate output of discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mu: means
        :param variances: tensor of diagonals of log_variances
        :param aux_labels_original: tensor of diagonals of log_variances
        :param aux_labels_predicted: tensor of diagonals of log_variances
        :return:
        """

        # reconstruction errors, not used as part of loss just to monitor
        nle = 0.5 * (ten_original.view(len(ten_original), -1)) - ten_predict.view((len(ten_predict), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mu, 2) + variances + 1, 1)

        # mse between intermediate layers
        mse = torch.sum((layer_original - layer_predicted) ** 2, 1)

        # BCE for decoder & discriminator for original, sampled & reconstructed
        # the only excluded is the bce_gen original

        bce_dis_original = -torch.log(labels_original)
        bce_dis_sampled = -torch.log(1 - labels_sampled)

        bce_gen_original = -torch.log(1 - labels_original)
        bce_gen_sampled = -torch.log(labels_sampled)

        aux_criteron = nn.NLLLoss()
        nllloss_aux_original = aux_criteron(aux_labels_predicted, aux_labels_original)
        nllloss_aux_sampled = aux_criteron(aux_labels_sampled, aux_labels_original)

        '''
        bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                                        Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        '''

        return nle, kl, mse, bce_dis_original, bce_dis_sampled, bce_gen_original, bce_gen_sampled,\
            nllloss_aux_original, nllloss_aux_sampled
