"""Unit tests for EGNN v2 implementation."""
import unittest
import torch
from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn_v2 import EGNN_V2
from fast_td3.actors.actor_egnn_v2 import ActorEGNN_V2


class TestGraphBuilder(unittest.TestCase):
    """Test graph builder functionality."""

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
        """Test that generate_input returns correct shapes for env without object."""
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

    def test_generate_input_for_mixed_type_shapes(self):
        """Test that generate_input_for_mixed_type returns correct shapes."""
        builder = GraphBuilder(
            env_name="h1-balance_simple-v0",
            batch_size=self.batch_size,
            device=self.device,
            robot="h1"
        )
        
        obs = torch.randn(self.batch_size, 64)
        xanchor = torch.randn(self.batch_size, 21, 3)
        
        h_node, h_object, x_joint, x_object = builder.generate_input_for_mixed_type(obs, xanchor)
        
        # Check joint features shape [batch*num_joints, 2]
        self.assertEqual(h_node.shape, (self.batch_size * 19, 2))
        
        # Check object features shape [batch, 26]
        self.assertEqual(h_object.shape, (self.batch_size, 26))
        
        # Check joint coordinates shape [batch*num_joints, 3]
        self.assertEqual(x_joint.shape, (self.batch_size * 19, 3))
        
        # Check object coordinates shape [batch, 3]
        self.assertEqual(x_object.shape, (self.batch_size, 3))


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

    def test_egnn_v2_mixed_type_forward(self):
        """Test EGNN_V2 forward pass for environments with objects."""
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=0,  # Unused due to LazyLinear
            hidden_nf=self.hidden_dim,
            out_node_nf=1,
            in_edge_nf=0,
            device=self.device,
            batch_size=self.batch_size,
            act_fn=torch.nn.SiLU(),
            n_layers=2,
            robot="h1",
            env_name="h1-balance_simple-v0",  # Environment with object
        )
        
        # Create mock observations for balance_simple (64 dims)
        obs = torch.randn(self.batch_size, 64)
        xanchor = torch.randn(self.batch_size, 21, 3)  # 21 positions for env with object
        
        # Forward pass
        output = model(obs, xanchor)
        
        # Output should be [batch_size, num_joints]
        self.assertEqual(output.shape, (self.batch_size, 19))
        
        # Output should be bounded by tanh
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))


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
        actor = ActorEGNN_V2(
            num_envs=self.batch_size,  # Match num_envs with batch_size for noise shape compatibility
            hidden_dim=64,
            batch_size=self.batch_size,
            device=self.device,
            n_layers=2,
            act_fn="silu",
            env_name="h1-stand-v0",
            robot="h1",
            std_min=0.1,
            std_max=0.5,
        )
        
        obs = torch.randn(self.batch_size, 51)
        xanchor = torch.randn(self.batch_size, 20, 3)
        
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

    def test_actor_egnn_v2_balance_simple(self):
        """Test ActorEGNN_V2 with h1-balance_simple-v0 environment."""
        actor = ActorEGNN_V2(
            num_envs=self.num_envs,
            hidden_dim=64,
            batch_size=self.batch_size,
            device=self.device,
            n_layers=2,
            act_fn="silu",
            env_name="h1-balance_simple-v0",
            robot="h1",
        )
        
        # balance_simple has 64-dim observations and 21 xanchor positions
        obs = torch.randn(self.batch_size, 64)
        xanchor = torch.randn(self.batch_size, 21, 3)
        
        output = actor(obs, xanchor)
        
        # Output should be [batch_size, num_joints=19]
        self.assertEqual(output.shape, (self.batch_size, 19))
        
        # Actions should be in [-1, 1] due to Tanh
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))


class TestEGNN_V2_NodeIndexing(unittest.TestCase):
    """Test that node indexing follows the specification."""

    def test_node_indexing_pattern(self):
        """
        Test that joint edge indices follow the pattern:
        - Batch 0 joints: indices 0-18
        - Batch 1 joints: indices 19-37
        """
        device = torch.device("cpu")
        batch_size = 2
        num_joints = 19
        
        model = EGNN_V2(
            in_joint_nf=2,
            in_object_nf=6,
            hidden_nf=32,
            out_node_nf=1,
            in_edge_nf=0,
            device=device,
            batch_size=batch_size,
            act_fn=torch.nn.SiLU(),
            n_layers=1,
            robot="h1",
            env_name="h1-stand-v0",
        )
        
        # Generate joint edge indices
        edges = model.generate_joint_edges(batch_size, device)
        
        # Check that all edge indices are within the joint node range
        # Joints for batch 0: 0-18, batch 1: 19-37
        max_joint_idx = batch_size * num_joints - 1
        self.assertTrue(torch.all(edges[0] <= max_joint_idx))
        self.assertTrue(torch.all(edges[1] <= max_joint_idx))
        
        # Verify that batch 0 joint edges are in range [0, 18]
        batch_0_edges = edges[:, edges[0] < num_joints]
        self.assertTrue(torch.all(batch_0_edges[0] < num_joints))
        
        # Verify that batch 1 joint edges are in range [19, 37]
        batch_1_edges = edges[:, edges[0] >= num_joints]
        self.assertTrue(torch.all(batch_1_edges[0] >= num_joints))
        self.assertTrue(torch.all(batch_1_edges[0] < 2 * num_joints))


if __name__ == "__main__":
    unittest.main()
