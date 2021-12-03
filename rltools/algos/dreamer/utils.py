import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import clear_output
from rltools.transformations.transformations import from_numpy, to_torch, \
    SliceTrajectory, TrajToDevice, to_numpy

import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# TODO fix warnings rather than suppressing them


@torch.no_grad()
def rollout(env, wm):
    obs = env.reset()
    wm.reset()

    observations, rewards, dones = [], [], []
    done = False
    tr = defaultdict(list)
    while not done:
        resp = wm.act(obs)
        for k, v in resp.items():
            v = v.detach().cpu().flatten().numpy()
            resp[k] = v
            tr[k].append(v)
        next_obs, reward, done, _ = env.step(resp['actions'])
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        if done:
            break
        obs = next_obs

    tr.update(
        observations=observations,
        rewards=rewards,
        done_flags=dones)

    tr = from_numpy(to_numpy(tr))
    return tr


def rnd_rollout(env, steps=10**2):
    obs = env.reset()
    observations, actions, rewards, dones = [], [], [], []
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        if done:
            break
        obs = next_obs
    return to_torch({
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'done_flags':dones
           })


def populate_buffer(env, wm, buffer, to_device, slicer, n_epochs=50):
    logs = []
    for _ in range(n_epochs):
        tr = to_device(rnd_rollout(env))
        logs.append(wm.dynamics_learning(tr).detach().cpu().numpy())
        buffer.add(slicer(tr))


def update_buffer(env, wm, buffer, slicer):
    wm.train()
    tr = rollout(env, wm)
    buffer.add(slicer(tr))


def sb_dataloader(buffer):
    dl = DataLoader(buffer, shuffle=True)
    for item in dl:
        for k, v in item.items():
            item[k] = v.squeeze(0)
        yield item


def train_from_buffer(wm, buffer, to_device):
    dl = sb_dataloader(buffer)
    loss = 0
    wm.train()
    for idx, tr in enumerate(dl):
        loss += wm.dynamics_learning(to_device(tr))
        break
    return loss


@torch.no_grad()
def evaluate(env, wm, n=3, plot_rewards=True):
    wm.eval()
    stats = []
    for i in range(n):
        tr = rollout(env, wm)
        stats.append(tr['rewards'].sum())
        if plot_rewards:
            clear_output(wait=True)
            plt.plot(tr['rewards'].flatten().tolist())
            plt.title('Epoch= '+str(i))
            plt.show()
    wm.train()
    return np.mean(stats), np.std(stats)


def train(env, wm, buffer, n_updates=30, n_evals=3, n_slices=40):
    logs = []
    wm_losses = []
    S = SliceTrajectory(n_slices)
    T = TrajToDevice(wm.device)

    populate_buffer(env, wm, buffer, T, S, n_epochs=20)

    while True:

        update_buffer(env, wm, buffer, S)

        wm_loss = []
        for _ in range(n_updates):
            loss = train_from_buffer(wm, buffer, T)
            wm_loss.append(loss.item())
        wm_losses.append(np.mean(wm_loss))

        for _ in range(n_updates):
            tr = next(sb_dataloader(buffer))
            obs = tr['observations'][0]
            wm.behavior_learning(obs)

        clear_output(wait=True)
        #plt.plot(wm_losses)
        #plt.show()
        logs.append(evaluate(env, wm, n_evals))
        mean, std = map(np.array, zip(*logs))
        plt.plot(mean)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=.1)
        plt.show()
