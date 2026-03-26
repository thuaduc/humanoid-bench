import unittest
import numpy as np
import torch
from humanoid_bench.envs.custom_env import Stand, BalanceSimple, SitHard, unflatten_obs, get_obs_shapes


class TestCustomEnv(unittest.TestCase):

    def test_stand_get_obs_shapes(self):
        shapes = Stand.get_obs_shapes()
        expected = {
            "object_features": (13,),
            "joint_features": (19, 5),
        }
        self.assertEqual(shapes, expected)

    def test_balance_simple_get_obs_shapes(self):
        shapes = BalanceSimple.get_obs_shapes()
        expected = {
            "object_features": (26,),
            "joint_features": (19, 5),
        }
        self.assertEqual(shapes, expected)

    def test_sit_hard_get_obs_shapes(self):
        shapes = SitHard.get_obs_shapes()
        expected = {
            "object_features": (23,),
            "joint_features": (19, 5),
        }
        self.assertEqual(shapes, expected)

    def test_stand_unflatten_obs(self):
        batch_size = 2
        obs_dim = 108  # 13 + 19*5
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, h_objects = Stand.unflatten_obs(flat_obs)
        self.assertEqual(h_joints.shape, (batch_size, 19, 5))
        self.assertEqual(h_objects.shape, (batch_size, 1, 13))

    def test_balance_simple_unflatten_obs(self):
        batch_size = 2
        obs_dim = 121  # 26 + 19*5
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, h_objects = BalanceSimple.unflatten_obs(flat_obs)
        self.assertEqual(h_joints.shape, (batch_size, 19, 5))
        self.assertEqual(h_objects.shape, (batch_size, 1, 26))

    def test_sit_hard_unflatten_obs(self):
        batch_size = 2
        obs_dim = 118  # 23 + 19*5
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, h_objects = SitHard.unflatten_obs(flat_obs)
        self.assertEqual(h_joints.shape, (batch_size, 19, 5))
        self.assertEqual(h_objects.shape, (batch_size, 1, 23))

    def test_unflatten_preserves_values(self):
        """Verify that view() correctly maps node features without copying."""
        batch_size = 1
        obs_dim = 108
        flat_obs = torch.arange(obs_dim, dtype=torch.float32).unsqueeze(0)
        h_joints, h_objects = Stand.unflatten_obs(flat_obs)
        # h_objects should be flat_obs[:, :13]
        self.assertTrue(torch.allclose(h_objects.squeeze(1), flat_obs[:, :13]))
        # h_joints[0, node_i, feature_j] == flat_obs[0, 13 + node_i*5 + feature_j]
        for node_i in range(19):
            for feat_j in range(5):
                expected = flat_obs[0, 13 + node_i * 5 + feat_j].item()
                actual = h_joints[0, node_i, feat_j].item()
                self.assertAlmostEqual(actual, expected)

    def test_global_get_obs_shapes(self):
        shapes = get_obs_shapes("h1-stand-v1")
        expected = {
            "object_features": (13,),
            "joint_features": (19, 5),
        }
        self.assertEqual(shapes, expected)

    def test_global_unflatten_obs(self):
        batch_size = 1
        obs_dim = 108
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, h_objects = unflatten_obs(flat_obs, "h1-stand-v1")
        self.assertEqual(h_joints.shape, (batch_size, 19, 5))
        self.assertEqual(h_objects.shape, (batch_size, 1, 13))


if __name__ == '__main__':
    unittest.main()
