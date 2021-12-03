import torch
import torch.nn as nn
from rltools.common.models import NormalModel
from .utils import LossSink
from rltools.common.utils import NormalNormalKL


class RSSMCell(nn.Module):
    kl_coef = 1e0
    lr, eps = 6e-4, 1e-5

    def __init__(self,
                 encoded_observation_size,
                 action_size,
                 hidden_size=10,
                 latent_size=4,
                 encoder=nn.Identity(),
                 decoder=nn.Identity(),
                 ):

        super().__init__()

        kwargs = dict(hidden_sizes=3*[64])

        self.hidden_size, self.latent_size, self.action_size, self.obs_size = \
            hidden_size, latent_size, action_size, encoded_observation_size

        self.encoder = encoder
        self.decoder = decoder

        self.transition_model = NormalModel(hidden_size, latent_size, **kwargs)  # p(s_t | h_t)

        self.reward_model = NormalModel(hidden_size + latent_size, 1, **kwargs)  # p(r_t | h_t, s_t)

        self.observation_model = nn.Sequential(
            NormalModel(hidden_size + latent_size, encoded_observation_size, **kwargs),
            decoder
        )  # p(o_t | h_t, s_t)

        self.hidden_model = nn.GRUCell(latent_size + action_size, hidden_size)  # f(h_{t-1}, s_{t-1}, a_{t-1})

        self.inference_model = NormalModel(hidden_size + encoded_observation_size, latent_size, **kwargs)  # q(s_t | h_t, o_t)

        self.device = 'cpu'
        self.optim = None
        self.h = self.init_hidden()
        self.ls = LossSink()

    def step(self, obs, reward, prev_state=None, prev_action=None, prev_h=None):  # o_t, r_t, s_{t-1}, a_{t-1}, h_{t-1}

        prev_state = self._fill_on_empty(prev_state, (obs.shape[0], self.latent_size))
        prev_action = self._fill_on_empty(prev_action, (obs.shape[0], self.action_size))
        prev_h = self._fill_on_empty(prev_h, (obs.shape[0], self.hidden_size))

        h = self.hidden_model(torch.cat((prev_state, prev_action), -1), prev_h)

        state_dist = self.transition_model.dist(h)

        inference_dist = self.infer(obs, h)

        # kl = torch.clamp(kl, min=3.)  # 3 free nats as in the paper
        state = inference_dist.rsample()

        if self.training:
            hs = torch.cat((h, state), -1)
            obs_gen = self.observation_model(hs)
            reward_dist = self.reward_model.dist(hs)

            obs_loss = (obs - obs_gen).pow(2).mean()
            rew_log_prob = reward_dist.log_prob(reward).mean()
            kl = NormalNormalKL(inference_dist, state_dist).mean()
            self.ls += obs_loss - rew_log_prob + self.kl_coef * kl

        return state, h

    def learn(self, trajectory: dict):
        self.train()
        # assert trajectory['actions'].device == self.device
        observations, actions, rewards = map(lambda key: trajectory[key].unsqueeze(1),
                                             ('observations', 'actions', 'rewards'))
        rewards = rewards.unsqueeze(-1)
        # assert rewards.ndimension() == 3, "(batch, 1, 1)"
        # assert observations.ndimension() == 5, "(batch, 1, 3, 64, 64)"
        # assert actions.ndimension() == 3, "(batch, 1, 1)"
        self.zero_grad()
        prev_h = self.init_hidden()
        prev_action = actions[0]

        with torch.no_grad():
            prev_state = self.infer(observations[0], prev_h).sample()

        for i, (obs, action, reward) in enumerate(zip(observations[1:], actions[1:], rewards[1:])):
            prev_state, prev_h = self.step(obs, reward, prev_state, prev_action, prev_h)
            prev_action = action

        self.ls.backward()
        self.optim.step()
        total_loss = self.ls()

        return total_loss

    def rollout(self, agent, seed_state, horizon=10, from_obs=False):
        h = self.h  # self.h should hold smth similar to h_t when method is called
        if from_obs:
            seed_obs = self.to_tensor(seed_state)
            state = self.infer(seed_obs, h).rsample()
        else:
            state = seed_state
        states, actions, rewards = [], [], []

        for _ in range(horizon):
            action = agent.act(state.flatten())['actions']
            assert action.ndimension() == 1 and action.requires_grad is True
            action = self.to_tensor(action)  # a_t
            reward = self.reward_model(torch.cat((h, state), -1))  # r_t
            h = self.hidden_model(torch.cat((state, action), -1), h)  # h_{t+1}
            states.append(state.flatten())
            rewards.append(reward.flatten())
            actions.append(action.flatten())
            state = self.transition_model(h)  # s_{t+1}

        states = torch.stack(states)

        return {
            'states': states,
            'actions': torch.stack(actions),
            'rewards': torch.stack(rewards),
            'done_flags': torch.cat((torch.zeros((states.shape[0] - 1, 1)), torch.tensor([[1.]]))).to(states.device)
        }

    def zero_grad(self):
        super().zero_grad()
        self.ls.zero_grad()
        return self

    def update_hidden(self, state, action):  # the only way to update h_t -> h_{t+1}
        self.h = self.hidden_model(torch.cat((state, action), -1), self.h)

    def init_hidden(self):
        # can be called through _fill_on_empty
        return torch.zeros((1, self.hidden_size), device=self.device, requires_grad=True)

    def reset(self):
        self.h = self.init_hidden()
        return self

    def to_tensor(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.FloatTensor(inp)
        inp = inp.unsqueeze(0).to(self.device)
        return inp

    def infer(self, obs, h=None) -> torch.distributions.Distribution:
        if h is None:
            h = self.h
        obs_enc = self.encoder(obs)
        return self.inference_model.dist(torch.cat((h,  obs_enc), -1))

    def _fill_on_empty(self, tensor, shape):
        return torch.zeros(shape, requires_grad=True, device=self.device) if tensor is None else tensor

    def compile(self, device='cpu', optim=None):
        # self.apply() #init
        self.device = device
        self.to(device)
        self.h = self.init_hidden()
        self.optim = optim or torch.optim.Adam(self.parameters(), self.lr, eps=self.eps)
        return self
