import torch
import torch.nn as nn
from rltools.models.rssm import RSSMCell
from rltools.transformations import MergeTrajectories
from .models import Policy, ValueNet
from .utils import train
from rltools.common.buffers import SliceBuffer


class Dreamer(nn.Module):
    def __init__(self, obs_size,
                 action_size,
                 hidden_size,
                 latent_size,
                 encoder=nn.Identity(),
                 decoder=nn.Identity(),
                 ):
        super().__init__()

        self.rssm = RSSMCell(obs_size, action_size, hidden_size, latent_size,
                             encoder=encoder, decoder=decoder)
        self.policy = Policy(latent_size, action_size, 3 * [100])
        self.V = ValueNet(latent_size, hidden_sizes=3 * [100])

        self.encoder = encoder
        self.decoder = decoder

        self.device = 'cpu'

    # @torch.no_grad() #it may block required gradient propagation
    # actually gradient's not needed but still
    def act(self, obs):
        obs = self.rssm.to_tensor(obs)
        state = self.rssm.infer(obs).rsample()
        resp = self.policy.act(state)
        action = resp['actions']
        self.rssm.update_hidden(state, action.unsqueeze(0))
        return resp

    def dynamics_learning(self, trajectory):
        wm_loss = self.rssm.learn(trajectory)
        return wm_loss

    def behavior_learning(self, seed_obs, horizon=35):
        self.reset()
        tr_list = []
        for _ in range(5):
            tr = self.rssm.rollout(self.policy, seed_obs, horizon=horizon, from_obs=True)
            tr['values'] = self.V(tr['states']).flatten()
            tr['target_values'] = self._target_values(tr)
            tr_list.append(tr)
        tr = MergeTrajectories()(tr_list)
        pl = self.policy.learn(tr)
        vl = self.V.learn(tr)
        return vl + pl

    def reset(self):
        self.rssm.reset()
        return self

    def _target_values(self, trajectory, gamma=.99):
        # TODO redo target_values computation as it is in the paper
        rewards = trajectory['rewards']
        v = self.V(trajectory['states'][-1]).flatten()
        targets = v
        for r in rewards.flip(0):
            v = gamma * v + r
            targets = torch.cat((targets, v))
        return targets.flip(0)[:-1]

    def compile(self, device='cpu', optim=None):
        self.to(device)
        self.rssm.compile(device=device)
        self.policy.compile(device=device)
        self.V.compile(device=device)
        # possibly torch have automatic realization of chain call of this type
        self.device = device

    def learn(self, env):
        self.buffer = SliceBuffer(10**3)
        train(env, self, self.buffer, n_updates=60, n_evals=5, n_slices=40)
