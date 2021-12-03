import numpy as np


class SliceBuffer:
    def __init__(self, maxsize=10**2):
        self.maxsize = maxsize
        self._data = []
        self._t = 0

    def __getitem__(self, idx):
        return self._data[idx]

    def add(self, slices: list):
        if len(self._data) > self.maxsize:
            for sl in slices:
                self._data[self._t] = sl
                self._t = (self._t + 1) % self.maxsize
        else:
            self._data.extend(slices)

    def __len__(self):
        return self._data.__len__()


class ReplayBuffer:
    def __init__(self, size=10 ** 2):
        self._size = size
        self._data = []
        self._t = 0
        self._ready = False

    def sample(self, size):
        ind = np.random.randint(len(self), size=size)
        sample = [self._data[i] for i in ind]
        return list(zip(*sample))

    def __getitem__(self, ind):
        return self._data[ind]

    def __len__(self):
        return len(self._data)

    def add(self, transitions):
        #         if not isinstance(transitions[0], Iterable):
        #             transitions = [transitions]
        for transition in transitions:
            if len(self) < self.size:
                self._data.append(transition)
            else:
                self._data[self._t] = transition
            self._t = (self._t + 1) % self.size

    @property
    def size(self):
        return self._size

    @property
    def is_ready(self):
        return len(self) == self.size
