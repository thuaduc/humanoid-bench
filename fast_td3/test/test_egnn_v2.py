"""Unit tests for EGNN v2 implementation."""
import unittest
import torch
from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn_v2 import EGNN_V2
from fast_td3.actors.actor_egnn_v2 import ActorEGNN_V2


class TestGraphBuilderV2(unittest.TestCase):
    """Test graph builder v2 functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.builder = GraphBuilder(
            env_name="h1-stand-v0",
            batch_size=self.batch_size,
            device=self.device,
            robot="h1"
        )

    def test_generate_input_output_shapes(self):
        """Test that generate_input returns correct shapes."""
        # Create mock observations and xanchor
        obs = torch.randn(self.batch_size, 51)
        xanchor = torch.randn(self.batch_size, 20, 3)
        
        h_joints, x_joints, h_objects, x_objects = self.builder.generate_input(obs, xanchor)
        
        # Check joint features shape [batch*num_joints, 2]
        self.assertEqual(h_joints.shape, (self.batch_size * 19, 2))
        
        # Check joint coordinates shape [batch*num_joints, 3]
        self.assertEqual(x_joints.shape, (self.batch_size * 19, 3))
        
        # Check object features shape [batch, 6]
        self.assertEqual(h_objects.shape, (self.batch_size, 6))
        
        # Check object coordinates shape [batch, 3]
        self.assertEqual(x_objects.shape, (self.batch_size, 3))

    def test_generate_input_object_shapes(self):
        """Test that generate_input_object returns correct shapes for object environments."""
        # Create builder for object environment
        builder_obj = GraphBuilder(
            env_name="h1-balance_simple-v0",
            batch_size=self.batch_size,
            device=self.device,
            robot="h1"
        )
        
        obs = torch.randn(self.batch_size, 64)
        xanchor = torch.randn(self.batch_size, 21, 3)
        
        h_joints, x_joints, h_objects, x_objects = builder_obj.generate_input_object(obs, xanchor)
        
        # Check joint features shape [batch*num_joints, 2]
        self.assertEqual(h_joints.shape, (self.batch_size * 19, 2))
        
        # Check joint coordinates shape [batch*num_joints, 3]
        self.assertEqual(x_joints.shape, (self.batch_size * 19, 3))
        
        # Check object features shape [batch*2, 10] - 2 objects (root + object), 10 features each
        self.assertEqual(h_objects.shape, (self.batch_size * 2, 10))
        
        # Check object coordinates shape [batch*2, 3]
        self.assertEqual(x_objects.shape, (self.batch_size * 2, 3))


class TestEGNN_V2(unittest.TestCase):
    """Test EGNN v2 model."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.hidden_dim = 64
        self.n_layers = 3
        
    def test_egnn_v2_creation(self):
        """Test that EGNN_V2 can be created."""
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=6,
            hidden_nf=self.hidden_dim,
            out_node_nf=1,
            in_edge_nf=0,
            device=self.device,
            batch_size=self.batch_size,
            act_fn=torch.nn.SiLU(),
            n_layers=self.n_layers,
            robot="h1",
            env_name="h1-stand-v0",
        )
        self.assertIsNotNone(model)

    def test_egnn_v2_forward_shapes(self):
        """Test that EGNN_V2 forward pass produces correct output shapes."""
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=6,
            hidden_nf=self.hidden_dim,
            out_node_nf=1,
            in_edge_nf=0,
            device=self.device,
            batch_size=self.batch_size,
            act_fn=torch.nn.SiLU(),
            n_layers=self.n_layers,
            robot="h1",
            env_name="h1-stand-v0",
        )
        
        # Create mock observations
        obs = torch.randn(self.batch_size, 51)
        xanchor = torch.randn(self.batch_size, 20, 3)
        
        # Forward pass
        output = model(obs, xanchor)
        
        # Output should be [batch_size, num_joints]
        self.assertEqual(output.shape, (self.batch_size, 19))

    def test_egnn_v2_different_batch_sizes(self):
        """Test that EGNN_V2 handles different batch sizes correctly."""
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=6,
            hidden_nf=self.hidden_dim,
            out_node_nf=1,
            in_edge_nf=0,
            device=self.device,
            batch_size=4,  # Initial batch size
            act_fn=torch.nn.SiLU(),
            n_layers=2,
            robot="h1",
            env_name="h1-stand-v0",
        )
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            obs = torch.randn(batch_size, 51)
            xanchor = torch.randn(batch_size, 20, 3)
            
            output = model(obs, xanchor)
            self.assertEqual(output.shape, (batch_size, 19))

    def test_egnn_v2_cross_edge_generation(self):
        """Test that cross-graph edge generation is correct."""
        batch_size = 3
        num_joints = 19
        num_objs = 2
        
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=10,
            hidden_nf=32,
            out_node_nf=1,
            in_edge_nf=0,
            device=self.device,
            batch_size=batch_size,
            act_fn=torch.nn.SiLU(),
            n_layers=1,
            robot="h1",
            env_name="h1-balance_simple-v0",
        )
        
        cross_edges = model.generate_cross_edges(batch_size, num_objs, self.device)
        
        # Total edges: batch_size * num_joints * num_objs
        expected_num_edges = batch_size * num_joints * num_objs
        self.assertEqual(cross_edges.shape[1], expected_num_edges)
        
        # Check each batch's joints connect to the correct objects
        object_start_idx = batch_size * num_joints
        
        for b in range(batch_size):
            joint_start = b * num_joints
            joint_end = (b + 1) * num_joints
            obj_start = object_start_idx + b * num_objs
            obj_end = obj_start + num_objs
            
            # Get edges for this batch's joints
            mask = (cross_edges[0] >= joint_start) & (cross_edges[0] < joint_end)
            batch_edges = cross_edges[:, mask]
            
            # Verify source indices
            self.assertEqual(batch_edges[0].min().item(), joint_start)
            self.assertEqual(batch_edges[0].max().item(), joint_end - 1)
            
            # Verify destination indices (should be this batch's objects)
            unique_dsts = sorted(batch_edges[1].unique().tolist())
            expected_dsts = list(range(obj_start, obj_end))
            self.assertEqual(unique_dsts, expected_dsts)


class TestActorEGNN_V2(unittest.TestCase):
    """Test ActorEGNN_V2."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_envs = 4
        
    def test_actor_egnn_v2_creation(self):
        """Test that ActorEGNN_V2 can be created."""
        actor = ActorEGNN_V2(
            num_envs=self.num_envs,
            hidden_dim=64,
            batch_size=self.batch_size,
            device=self.device,
            n_layers=2,
            act_fn="silu",
            env_name="h1-stand-v0",
            robot="h1",
        )
        self.assertIsNotNone(actor)

    def test_actor_egnn_v2_forward(self):
        """Test that ActorEGNN_V2 forward pass works."""
        actor = ActorEGNN_V2(
            num_envs=self.num_envs,
            hidden_dim=64,
            batch_size=self.batch_size,
            device=self.device,
            n_layers=2,
            act_fn="silu",
            env_name="h1-stand-v0",
            robot="h1",
        )
        
        obs = torch.randn(self.batch_size, 51)
        xanchor = torch.randn(self.batch_size, 20, 3)
        
        output = actor(obs, xanchor)
        
        # Output should be [batch_size, num_joints=19]
        self.assertEqual(output.shape, (self.batch_size, 19))
        
        # Actions should be in [-1, 1] due to Tanh
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_actor_egnn_v2_explore(self):
        """Test that ActorEGNN_V2 explore method adds noise."""
        # Use num_envs as batch_size for this test since noise_scales is sized for num_envs
        actor = ActorEGNN_V2(
            num_envs=self.num_envs,
            hidden_dim=64,
            batch_size=self.num_envs,  # Match batch_size with num_envs
            device=self.device,
            n_layers=2,
            act_fn="silu",
            env_name="h1-stand-v0",
            robot="h1",
            std_min=0.1,
            std_max=0.5,
        )
        
        obs = torch.randn(self.num_envs, 51)  # Use num_envs as batch size
        xanchor = torch.randn(self.num_envs, 20, 3)
        
        # Get deterministic action
        action_det = actor.explore(obs, xanchor, deterministic=True)
        
        # Get stochastic action
        action_stoch = actor.explore(obs, xanchor, deterministic=False)
        
        # They should be different (with very high probability)
        self.assertFalse(torch.allclose(action_det, action_stoch))

    def test_actor_egnn_v2_activation_functions(self):
        """Test that different activation functions work."""
        for act_fn in ["silu", "relu", "leaky_relu"]:
            actor = ActorEGNN_V2(
                num_envs=self.num_envs,
                hidden_dim=32,
                batch_size=self.batch_size,
                device=self.device,
                n_layers=1,
                act_fn=act_fn,
                env_name="h1-stand-v0",
                robot="h1",
            )
            
            obs = torch.randn(self.batch_size, 51)
            xanchor = torch.randn(self.batch_size, 20, 3)
            
            output = actor(obs, xanchor)
            self.assertEqual(output.shape, (self.batch_size, 19))


class TestEGNN_V2_NodeIndexing(unittest.TestCase):
    """Test that node indexing follows the specification."""

    def test_node_indexing_pattern(self):
        """
        Test that node indices follow the pattern:
        - Batch 0 joints: indices 0-18
        - Batch 1 joints: indices 19-37
        - Batch 0 objects: indices 38-39
        - Batch 1 objects: indices 40-41
        """
        device = torch.device("cpu")
        batch_size = 2
        num_joints = 19
        num_objs = 2
        
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=10,
            hidden_nf=32,
            out_node_nf=1,
            in_edge_nf=0,
            device=device,
            batch_size=batch_size,
            act_fn=torch.nn.SiLU(),
            n_layers=1,
            robot="h1",
            env_name="h1-balance_simple-v0",
        )
        
        # Generate joint edge indices
        joint_edges = model.generate_joint_edges(batch_size, device)
        
        # Check that all joint edge indices are within the joint node range
        # Joints for batch 0: 0-18, batch 1: 19-37
        max_joint_idx = batch_size * num_joints - 1
        self.assertTrue(torch.all(joint_edges[0] <= max_joint_idx))
        self.assertTrue(torch.all(joint_edges[1] <= max_joint_idx))
        
        # Verify that batch 0 joint edges are in range [0, 18]
        batch_0_edges = joint_edges[:, joint_edges[0] < num_joints]
        self.assertTrue(torch.all(batch_0_edges[0] < num_joints))
        
        # Verify that batch 1 joint edges are in range [19, 37]
        batch_1_edges = joint_edges[:, joint_edges[0] >= num_joints]
        self.assertTrue(torch.all(batch_1_edges[0] >= num_joints))
        self.assertTrue(torch.all(batch_1_edges[0] < 2 * num_joints))
        
        # Test cross edges - verify each batch connects to its own objects
        cross_edges = model.generate_cross_edges(batch_size, num_objs, device)
        object_start_idx = batch_size * num_joints  # 38
        
        # Batch 0 joints (0-18) should connect to objects 38-39
        batch_0_cross = cross_edges[:, cross_edges[0] < num_joints]
        self.assertEqual(sorted(batch_0_cross[1].unique().tolist()), [38, 39])
        
        # Batch 1 joints (19-37) should connect to objects 40-41
        batch_1_cross = cross_edges[:, cross_edges[0] >= num_joints]
        self.assertEqual(sorted(batch_1_cross[1].unique().tolist()), [40, 41])


if __name__ == "__main__":
    unittest.main()
