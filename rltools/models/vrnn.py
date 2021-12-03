import torch
import torch.nn as nn
from rltools.common.utils import NormalNormalKL
from rltools.common.models import NormalModel
from .utils import LossSink


class VRNNCell(nn.Module):

    def __init__(self, input_size,
                 latent_size,
                 hidden_size,
                 encoder=nn.Identity(),
                 decoder=nn.Identity(),
                 loss_sink=LossSink(),
                 kl_coef=1e0):
        super().__init__()
        self.input_size, self.latent_size, self.hidden_size = input_size, latent_size, hidden_size

        self.encoder = encoder;
        self.decoder = decoder

        self.prior_model = NormalModel(hidden_size, latent_size)  # 1. prior
        self.pxcz = NormalModel(latent_size + hidden_size, input_size)  # 2.generation
        self.hidden_model = nn.GRUCell(input_size + latent_size, hidden_size, bias=True)  # 3. recurrence
        self.qzcx = NormalModel(input_size + hidden_size, latent_size)  # 4. inference
        self.ls = loss_sink
        self.kl_coef = kl_coef

    def forward(self, x, prev_h=None):
        if prev_h is None:
            prev_h = torch.zeros((x.shape[0], self.hidden_size))

        z_prior_dist = self.prior_model(prev_h)
        x_enc = self.encoder(x)

        inf_dist = self.qzcx(torch.cat((x_enc, prev_h), -1))
        z_gen = inf_dist.rsample()

        x_dist = self.pxcz(torch.cat((z_gen, prev_h), -1))
        x_gen = x_dist.rsample()

        h = self.hidden_model(torch.cat((x_gen, z_gen), -1), prev_h)

        if self.training:
            x_dec = self.decoder(x_gen)
            # rec_loss = -x_dist.log_prob(x)
            rec_loss = (x_dec - x).pow(2).mean() / x.detach().max()
            kl_div = NormalNormalKL(inf_dist, z_prior_dist)
            self.ls += (rec_loss + self.kl_coef * kl_div).mean()
        return h


class VRNN(nn.Module):
    def __init__(self,
                 input_size,
                 latent_size,
                 hidden_size,
                 loss_sink,
                 num_layers=1,
                 batch_first=True,
                 kl_coef=1e-1
                 ):

        super().__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.input_size, self.latent_size, self.hidden_size = input_size, latent_size, hidden_size
        self.ls = loss_sink

        self.vcell = VRNNCell(input_size, latent_size, hidden_size, self.ls, kl_coef)
        self.hs = None

    def forward(self, X, hs=None):
        """
        N = batch_size, D = 1 + [bidir == True], L=lenght of sequence
        X: [L, N, input_size] if bf==false else [N,L, inp]
        h: [D*num_layerss, N, hidden_size]
        return: x=[N,L,D*hidden_size], h=[D*num_layers, N, hidden_size]
        """
        if self.batch_first:
            X = X.transpose(0, 1)
        if hs is None:
            self.hs = torch.zeros((self.num_layers, X.shape[1], self.hidden_size), requires_grad=True)
            hs = self.hs
        Hs = []  # Hs.shape = [L, num_layers, N, hid]
        output = []
        for x in X:
            new_hs = []
            for h in hs:
                new_h = self.vcell(x, h)
                new_hs.append(new_h)
            hs = torch.stack(new_hs)
            output.append(hs[-1].unsqueeze(0))
            Hs.append(hs)
        # TODO Make sure that RNN num_layers concept is as it is realized here
        #    --- Redo num_layers part
        output = torch.cat(output)
        if self.batch_first:
            output = output.transpose(0, 1)

        self.hs = hs

        return output, hs

    def zero_grad(self):
        super().zero_grad()
        self.ls.zero_grad()
        return self
