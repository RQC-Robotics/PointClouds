import torch
import torch.nn as nn
from torch.distributions import Normal
from rltools.common.models import FullyConnected
from rltools.common.utils import build_mlp


class Policy(FullyConnected):
    def __init__(self, input_size, output_size, hidden_sizes=[32]):
        super().__init__([1, 1])

        sizes = [input_size] + hidden_sizes + [output_size]

        self.mu = nn.Sequential(build_mlp(sizes), nn.Tanh())
        self.sigma = nn.Sequential(build_mlp(sizes), nn.Softplus(), nn.Tanh())
        self._epsilon = .3
        self.compile()

    def dist(self, states):
        mu = 5. * self.mu(states)
        sigma = self.sigma(states)
        dist = Normal(mu, sigma)
        return dist

    def forward(self, states):
        dist = self.dist(states)
        return dist.rsample()

    def act(self, state):
        action = self(state)
        action = action + self.epsilon * torch.randn_like(action)
        return {'actions': action.flatten()}

    @property
    def epsilon(self):
        return self._epsilon if self.training else 0.

    def learn(self, trajectory):
        values = trajectory['target_values']
        assert values.requires_grad
        loss = - torch.mean(values)
        return super().learn(loss, retain_graph=True, inputs=list(self.parameters()))


class ValueNet(FullyConnected):
    def __init__(self, input_size, hidden_sizes=[32, 32]):
        super().__init__([input_size] + hidden_sizes + [1])
        self.compile()

    def learn(self, trajectory):
        preds = self(trajectory['states']).flatten()
        targets = trajectory['target_values']
        loss = (preds - targets.detach().flatten()).pow(2).mean()
        return super().learn(loss, retain_graph=False, inputs=list(self.parameters()))


# as in the paper
encoder = nn.Sequential(
    nn.Conv2d(3, 32, 4, 2),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, 4, 2),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 128, 4, 2),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 256, 4, 2),
    nn.Flatten()
)

decoder = nn.Sequential(
    nn.Unflatten(-1, (1024, 1, 1)),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(1024, 128, 5, 2),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(128, 64, 5, 2),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(64, 32, 6, 2),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(32, 3, 6, 2),
    nn.ReLU(inplace=True),

)
