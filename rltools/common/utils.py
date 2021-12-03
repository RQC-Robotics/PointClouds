import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import defaultdict


def build_mlp(sizes, activation = nn.ReLU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(activation())
    return nn.Sequential(*mlp[:-1])


def NormalNormalKL(dist1, dist2):
    assert isinstance(dist1, Normal) and isinstance(dist2, Normal)
    mu1, var1 = dist1.mean, dist1.variance
    mu2, var2 = dist2.mean, dist2.variance
    return 0.5*(var1 / var2 + (mu1 - mu2)**2 / var2 + torch.log(var2) - torch.log(var1) - 1).sum(-1)


def xavier_init(p):
    if isinstance(p, nn.Linear):
        nn.init.xavier_normal_(p.weight)
        p.bias.data.fill_(0.)


class Profiler:
    def __init__(self):
        self._data = defaultdict(list)

    def update(self, key, value):
        self._data[key].append(value)

    def __getitem__(self, key):
        return self._data.get(key)

    def summary(self):
        return self._data

    def plot(self):
        for k, v in self._data.items():
            plt.title(k)
            plt.plot(v)
            plt.show()
