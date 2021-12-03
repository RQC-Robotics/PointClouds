# TODO this one should be replaced by more useful version
from torchvision.transforms import Resize
from PIL import Image
import numpy as np
from gym.spaces import Box
from abc import ABC


class Wrapper(ABC):
    """ Partially solves problem with  compatibality"""

    def __init__(self, env):
        self._env = env
        self.action_space = None
        self.action_space = self._infer_action_space(env)

    def observation(self, timestamp):
        return timestamp.observation

    def reward(self, timestamp):
        return timestamp.reward

    def done(self, timestamp):
        return timestamp.last()

    def step(self, action):
        timestamp = self._env.step(action)
        obs = self.observation(timestamp)
        r = self.reward(timestamp)
        d = self.done(timestamp)
        return obs, r, d, None

    def reset(self):
        return self.observation(self._env.reset())

    @staticmethod
    def _infer_action_space(env):
        spec = env.action_spec()
        return Box(low=spec.minimum.astype(np.float32), high=spec.maximum.astype(np.float32), shape=spec.shape)

    @property
    def unwrapped(self):
        if hasattr(self._env, 'unwrapped'):
            return self._env.unwrapped
        return self._env


class PixelsToGym(Wrapper):
    def __init__(self, env):
        self._env = env
        self.resize = Resize((64, 64))

    def observation(self, timestamp):
        obs = timestamp.observation['pixels']
        obs = Image.fromarray(obs, 'RGB')
        obs = self.resize(obs)
        obs = np.array(obs) / 255.
        return obs.transpose((2, 1, 0))

    @property
    def observation_space(self):
        # correspondent space have to be extracted from the dm_control API -> gym API
        return Box(low=0., high=1., shape=(3,))

    @property
    def action_space(self):
        return Box(low=-2., high=2., shape=(1,))


class dmWrapper(Wrapper):

    @staticmethod
    def observation(timestamp):
        return np.concatenate(list(timestamp.observation.values()))
