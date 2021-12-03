import numpy as np
import scipy.signal
import torch
from collections import defaultdict, deque


class Transformations:
    def __init__(self, transformations_list):
        self.T = transformations_list

    def __call__(self, trajectory):
        for T in self.T:
            trajectory = T(trajectory)
        return trajectory


def to_torch(trajectory):
    for k, v in trajectory.items():
        if not torch.is_tensor(v):
            trajectory[k] = torch.from_numpy(np.array(v).astype(np.float32))
    return trajectory


def to_numpy(trajectory):
    for k, v in trajectory.items():
        trajectory[k] = np.array(v).astype(np.float32)
    return trajectory


def from_numpy(trajectory):
    for k, v in trajectory.items():
        trajectory[k] = torch.from_numpy(np.ascontiguousarray(v).astype(np.float32))
    return trajectory


class GAE:
    def __init__(self, gamma, lam, policy):
        self.gamma = gamma
        self.lam = lam
        self.policy = policy  # agent goes here for value extraction, make sure it has .act() -> dict

    def __call__(self, trajectory):
        rewards = np.append(trajectory['rewards'], 0.)
        if self.policy:
            with torch.no_grad():
                lastval = self.policy(torch.tensor(trajectory['next_states'][-1])).max(-1)[0]  # .item()
        else:
            lastval = 0
        values = np.append(trajectory['values'], 0)  # torch.cat((trajectory['values'], lastval))
        # TODO: fix last_val calculation

        deltas = rewards[:-1] + self.gamma * (1. - trajectory['done_flags']) * values[1:] - values[:-1]
        trajectory['advantages'] = self.discount_cumsum(deltas)
        trajectory['target_values'] = values[:-1] + trajectory['advantages']
        return trajectory

    def discount_cumsum(self, x):
        return scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], x[::-1], axis=0)[::-1]


class AdvNorm:
    def __call__(self, trajectory):
        adv = trajectory['advantages']
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        trajectory['advantages'] = adv
        return trajectory


class DictToList:
    def __init__(self, keys: list = None):
        self.keys = keys or ['states', 'actions', 'rewards', 'next_states', 'done_flags']

    def __call__(self, trajectory):
        return [trajectory[k] for k in self.keys]


class ListToDict:
    def __init__(self, keys: list = None):
        self.keys = keys

    def __call__(self, transitions):
        trajectory = defaultdict(list)
        for item in zip(*transitions):
            for k, value in zip(self.keys, item):
                trajectory[k].append(value)
        for k, v in trajectory.items():
            trajectory[k] = torch.stack(v)
        return trajectory


class NStep:
    def __init__(self, N, gamma=1, keys=None):
        assert N > 1
        self.N = N
        self.gamma = gamma
        self.keys = keys or ['rewards', 'next_states', 'done_flags', 'values']

    def __call__(self, trajectory):
        partial_tr = defaultdict(list)
        l = len(trajectory['rewards'])
        nbuffer = {key: deque(maxlen=self.N) for key in self.keys}
        for i in reversed(range(l)):
            for k in self.keys:
                nbuffer[k].append(trajectory[k][i])
            cur_dict = self.deqToTransition(nbuffer)
            for k in self.keys:
                partial_tr[k].append(cur_dict[k])
        trajectory = {k: v[:1 - self.N] for k, v in trajectory.items()}
        trajectory.update({k: v[::-1][:1 - self.N] for k, v in partial_tr.items()})
        return trajectory

    def deqToTransition(self, nbuffer):
        l = len(nbuffer['rewards'])
        tmp_dict = defaultdict(int)
        for k in self.keys:
            tmp_dict[k] = nbuffer[k][0]
        for i in range(1, l):
            done = nbuffer['done_flags'][i]
            tmp_dict['rewards'] = nbuffer['rewards'][i] + self.gamma * (1. - done) * tmp_dict['rewards']
            if done:
                for k in self.keys:
                    if k == 'rewards':
                        continue
                    tmp_dict[k] = nbuffer[k][i]
        return tmp_dict


class TransformBatchedEnv:
    def __call__(self, trajectories):
        trajectory = defaultdict(list)
        for k, v in trajectories.items():
            if k == 'states':
                trajectory[k] = np.hstack(list(map(lambda x: x[:, None], trajectories[k])))
            else:
                trajectory[k] = np.vstack(v).T

        trajectories = [{k: v[i] for k, v in trajectory.items()} for i in range(trajectory['actions'].__len__())]
        return trajectories


class MergeTrajectories:

    def __call__(self, tr_list):
        keys = tr_list[0].keys()
        is_tensor = torch.is_tensor(tr_list[0]['actions'])
        tr = {}
        for k in keys:
            values = [trajectory[k] for trajectory in tr_list]
            if is_tensor:
                tr[k] = torch.cat(values)
            else:
                tr[k] = np.concatenate(values, 0).T
                if k == 'states':
                    tr[k] = tr[k].T
        return tr


class NextStates:
    def __call__(self, trajectory):
        states = trajectory['states']
        trajectory['next_states'] = np.roll(states, 1, axis=0)
        return trajectory


class SeparateByEpisods:
    def __call__(self, trajectory):
        tr_list = []
        sep = np.where(trajectory['done_flags'] == 1)[0]
        for i in range(len(sep)):
            tr_list.append({k: v[(sep[i - 1] + 1 if i > 0 else 0):sep[i] + 1] for k, v in trajectory.items()})
        return tr_list


class CummulativeReturns:
    def __init__(self, gamma):
        self.gamma = gamma
        # consider adding value net for versatility

    def __call__(self, trajectory):
        rewards = trajectory['rewards']
        dones = trajectory['done_flags']
        trajectory['target_values'] = self.extract_values(rewards, dones)
        return trajectory

    def extract_values(self, rewards, dones):
        values = []
        last = 0
        for r, d in zip(rewards[::-1], dones[::-1]):
            last = r + self.gamma * (1. - d) * last
            values.append(last)
        return np.array(values[::-1])


class TrajToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, trajectory):
        for k, v in trajectory.items():
            trajectory[k] = v.to(self.device)
        return trajectory


class SliceTrajectory:
    def __init__(self, size):
        self.size = size

    def __call__(self, trajectory):
        length = len(trajectory['actions'])
        count = np.ceil(length / self.size).astype(int)
        tr_slices = []
        for i in range(count):
            tr_slice = {k: v[i:i + self.size] for k, v in trajectory.items()}
            tr_slices.append(tr_slice)
        return tr_slices
