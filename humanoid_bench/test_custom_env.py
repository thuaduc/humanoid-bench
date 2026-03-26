import unittest
import numpy as np
import torch
from humanoid_bench.envs.custom_env import Stand, BalanceSimple, unflatten_obs, get_obs_shapes


class TestCustomEnv(unittest.TestCase):

    def test_stand_get_obs_shapes(self):
        shapes = Stand.get_obs_shapes()
        expected = {
            "object_features": (13,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }
        self.assertEqual(shapes, expected)

    def test_balance_simple_get_obs_shapes(self):
        shapes = BalanceSimple.get_obs_shapes()
        expected = {
            "object_features": (26,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }
        self.assertEqual(shapes, expected)

    def test_stand_unflatten_obs(self):
        batch_size = 2
        obs_dim = 108
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, x_joints, h_objects = Stand.unflatten_obs(flat_obs)
        self.assertEqual(h_joints.shape, (batch_size * 19, 2))
        self.assertEqual(x_joints.shape, (batch_size * 19, 3))
        self.assertEqual(h_objects.shape, (batch_size, 13))

    def test_balance_simple_unflatten_obs(self):
        batch_size = 2
        obs_dim = 121
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, x_joints, h_objects = BalanceSimple.unflatten_obs(flat_obs)
        self.assertEqual(h_joints.shape, (batch_size * 19, 2))
        self.assertEqual(x_joints.shape, (batch_size * 19, 3))
        self.assertEqual(h_objects.shape, (batch_size, 26))

    def test_global_get_obs_shapes(self):
        shapes = get_obs_shapes("h1-stand-v1")
        expected = {
            "object_features": (13,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }
        self.assertEqual(shapes, expected)

    def test_global_unflatten_obs(self):
        batch_size = 1
        obs_dim = 108
        flat_obs = torch.randn(batch_size, obs_dim)
        h_joints, x_joints, h_objects = unflatten_obs(flat_obs, "h1-stand-v1")
        self.assertEqual(h_joints.shape, (batch_size * 19, 2))
        self.assertEqual(x_joints.shape, (batch_size * 19, 3))
        self.assertEqual(h_objects.shape, (batch_size, 13))


if __name__ == '__main__':
    unittest.main()