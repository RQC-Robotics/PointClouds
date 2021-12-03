import unittest
from rltools import transformations
import numpy as np
import torch
from itertools import accumulate


class TrajectoriesTreansformations(unittest.TestCase):

    def setUp(self):
        self.trajectory = {
            'actions': np.random.randint(5, size=10),
            'states': np.random.randn(10, 3),
            'rewards': list(range(10)),
            'done_flags': np.array(9*[False]+[True])
        }

    def test_types_changes(self):
        result = transformations.from_numpy(self.trajectory)
        self.assertTrue(torch.is_tensor(result['done_flags']))
        result = transformations.to_numpy(result)
        self.assertTrue(all([type(v) == np.ndarray for v in result.values()]))
        result = transformations.to_torch(result)
        self.assertTrue(all([torch.is_tensor(v) for v in result.values()]))

    def test_cum_sum(self):
        T = transformations.CummulativeReturns(gamma=1.)
        result = T(self.trajectory)
        print(result['target_values'], list(accumulate(self.trajectory['rewards']))[::-1])
        self.assertAlmostEqual(result['target_values'], list(accumulate(self.trajectory['rewards'][::-1])))
        self.test_keys(result)

    def test_keys(self, tr):
        possible_keys = set('actions', 'states', 'done_flags', 'rewards', 'target_values', 'values', 'advantages')
        self.assertTrue(all([k in possible_keys for k in tr.keys()]))


if __name__ == '__main__':
    unittest.main()
