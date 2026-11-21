import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'Install torch to exercise motion tensor kernels')
class TemporalWeights(unittest.TestCase):
    def test_normalizes_over_observed_times_and_broadcasts_views(self):
        import torch
        from freeforward.models.source_dynamics import temporal_source_weights
        ids = torch.tensor([[[[[1]], [[1]]], [[[1]], [[0]]], [[[0]], [[0]]]]])
        times = torch.tensor([[[0., 0.], [.5, .5], [1., 1.]]])
        weights = temporal_source_weights(ids, times, torch.tensor([[.3]]))
        self.assertAlmostEqual(float(weights[0, 0, 0] + weights[0, 0, 2]), 1., places=6)
        self.assertEqual(float(weights[0, 0, 0]), float(weights[0, 0, 1]))
        self.assertEqual(float(weights[0, 0, 3]), 1.)


if __name__ == '__main__':
    unittest.main()
