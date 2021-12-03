import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from rltools.common.utils import xavier_init, build_mlp
from torch.distributions import Normal


class FullyConnected(nn.Module):

    def __init__(self, layers_sizes: list, lr=1e-4, max_grad=1000.):
        super().__init__()

        self.model = build_mlp(layers_sizes)

        self._lr = lr
        self._max_grad = max_grad
        self.optim = None
        self.device = None
        self.apply(xavier_init)

    def forward(self, inp):
        return self.model(inp)

    def compile(self, device='cpu', optim=None):
        self.apply(xavier_init)
        self.optim = optim or torch.optim.Adam(self.parameters(), self._lr)
        self.device = device
        self.to(device)
        return self

    def learn(self, loss, *backward_args, **backward_kwargs):
        assert self.optim, "Compile first"
        self.optim.zero_grad()
        loss.backward(*backward_args, **backward_kwargs)
        if self._max_grad:
            clip_grad_norm_(self.parameters(), self._max_grad)
        self.optim.step()
        return loss.item()


class NormalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[32, 32]):
        super().__init__()

        sizes = [input_size] + hidden_sizes + [output_size]
        self.mu = nn.Sequential(build_mlp(sizes), nn.Tanh())
        self.sigma = nn.Sequential(
            build_mlp(sizes),
            nn.Softplus(),
            nn.Tanh()
        )

    def forward(self, inp):
        dist = self.dist(inp)
        return dist.rsample()

    def dist(self, inp):
        mu = 5.*self.mu(inp)
        sigma = self.sigma(inp)
        dist = Normal(mu, sigma)
        return dist

