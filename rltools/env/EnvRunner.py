import numpy as np
import torch
from collections import defaultdict, deque


class EnvRunner:
    first_transform = TransformBatchedEnv()

    def __init__(self, vec_env, agent, transformations: list = [], nsteps: int = 5):
        assert hasattr(agent, 'act'), 'Respect the required interface'
        self.agent = agent
        self._envs = vec_env
        self._states = self._envs.reset()
        self._interactions_count = 0
        self.transformations = transformations
        self.fin = (nsteps < 0)
        self.n_steps = int(10 ** 5) if self.fin else nsteps  # uncorrect behaviour with episodes in batching/merging
        # separate transformation suit for now

    def rollout(self):  # states ->self.states
        if self.fin:
            self._states = self._envs.reset()
            mask = np.zeros(self.nenvs, dtype=np.bool_)

        states_pool, rewards_pool, dones_pool, infos_pool = [], [], [], []
        trajectories = defaultdict(list)
        for i in range(self.n_steps):
            act_dict = self.agent.act(self._states)
            for k, v in act_dict.items():
                trajectories[k].append(v)
            next_states, rewards, dones, infos = self._envs.step(act_dict['actions'])
            states_pool.append(self._states)
            rewards_pool.append(rewards)
            dones_pool.append(dones)
            infos_pool.append(infos)
            self._states = next_states
            self._interactions_count += self.nenvs
            if self.fin:
                mask = np.bitwise_or(mask, dones)
                if np.all(mask):
                    break

        trajectories.update(
            states=states_pool,
            rewards=rewards_pool,
            done_flags=dones_pool,
            # infos=infos_pool
        )
        return trajectories

    @torch.no_grad()
    def __next__(self):
        trajectory = self.rollout()
        return self._apply_transformations(trajectory)

    def _apply_transformations(self, batched_trajectories):
        agents_trajectories = self.first_transform(batched_trajectories)
        trajectories = agents_trajectories.copy()
        for i, trajectory in enumerate(agents_trajectories):
            for transform in self.transformations:
                trajectory = transform(trajectory)
            trajectories[i] = trajectory

        return trajectories

    @torch.no_grad()
    def evaluate(self):
        logs = []
        self.agent.eval()
        for _ in range(10):
            mask = np.zeros(self.nenvs, dtype=np.bool_)
            states = self._envs.reset()
            Rs = np.zeros(self.nenvs)
            while not np.all(mask):
                actions = self.agent.act(states)['actions']
                states, rewards, dones, _ = self._envs.step(actions)
                Rs += rewards * (1 - mask)
                mask = np.bitwise_or(mask, dones)
            logs.extend(Rs.tolist())
        self.agent.train()
        self._states = self._envs.reset()
        return np.mean(logs), np.std(logs)

    @property
    def interactions_count(self):
        return self._interactions_count

    @property
    def nenvs(self):
        return self._envs.nenvs