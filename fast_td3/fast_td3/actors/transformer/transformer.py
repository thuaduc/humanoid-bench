import torch
import torch.nn as nn

from fast_td3.robots.graph_builder import GraphBuilder
from humanoid_bench.envs.custom_env import unflatten_obs


class Transformer(nn.Module):
    """
    Transformer-based actor for EGNN-style observations.
    
    Handles heterogeneous node types (joints and objects) with different feature dimensions
    by projecting them to a common embedding space before transformer processing.
    """

    def __init__(
        self,
        in_joint_nf,
        in_object_nf,
        out_node_nf,
        hidden_nf,
        device,
        batch_size,
        act_fn,
        n_layers,
        robot,
        env_name,
        num_heads=4,
        dropout=0.1,
    ):
        """
        :param in_joint_nf: Number of features for joint nodes (velocity + position) = 2
        :param in_object_nf: Number of features for object nodes = 13
        :param out_node_nf: Output dimension per node
        :param hidden_nf: Hidden dimension for embeddings and transformer
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param batch_size: Batch size for vectorized environments
        :param act_fn: Activation function
        :param n_layers: Number of transformer layers
        :param robot: Robot name for graph builder
        :param env_name: Environment name
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.batch_size = batch_size
        self.env_name = env_name
        self.robot = robot
        self.in_joint_nf = in_joint_nf
        self.in_object_nf = in_object_nf
        self.num_heads = num_heads
        
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        
        # Determine number of objects based on environment
        from fast_td3.actors.gnn.egcl import env_with_object
        self.num_objects = 2 if env_name in env_with_object else 1
        
        # Projection layers for heterogeneous features
        # Project joint features (2D) to hidden dimension
        self.joint_projection = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf),
            act_fn,
        )
        
        # Project object features (13D) to hidden dimension
        self.object_projection = nn.Sequential(
            nn.Linear(in_object_nf, hidden_nf),
            act_fn,
        )
        
        # Transformer encoder
        # Use divisor of hidden_nf for num_heads to avoid dimension issues
        actual_heads = min(num_heads, hidden_nf // 8)  # Ensure hidden_nf is divisible by num_heads
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_nf,
            nhead=actual_heads,
            dim_feedforward=hidden_nf * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layers,
        )
        
        # Output MLP: from transformer hidden features to actions
        # Total nodes = num_joints + num_objects (all with hidden_nf features)
        total_nodes = self.num_joints + self.num_objects
        self.output_mlp = nn.Sequential(
            nn.Linear(total_nodes * hidden_nf, hidden_nf * 4),
            act_fn,
            nn.Linear(hidden_nf * 4, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.num_joints),
            nn.Tanh(),
        )
        
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observations through the transformer.
        
        :param obs: Flattened observation tensor
        :return: Action tensor of shape (batch_size, num_joints)
        """
        obs = unflatten_obs(obs, self.env_name)
        current_batch_size = obs["joint_velocities"].shape[0]
        
        # Joint features: [velocity, position] for each joint
        joint_features = torch.cat([
            obs["joint_x"],
            obs["joint_velocities"].unsqueeze(-1),
            obs["joint_positions"].unsqueeze(-1),
        ], dim=-1).reshape(-1, self.in_joint_nf)  # Shape: (batch_size * num_joints, 5)
        
        # Object features: [position, quaternions, velocities] concatenated
        obj_feats = [
            obs["object_x"],
            obs["object_quaternions"],
            obs["object_velocities"],
        ]
        if "object_others" in obs:
            # Repeat for each object
            obj_others = obs["object_others"].unsqueeze(1).expand(-1, self.num_objects, -1)
            obj_feats.append(obj_others)

        object_features = torch.cat(obj_feats, dim=-1).reshape(-1, self.in_object_nf)  # Shape: (batch_size * num_objects, 13)
        
        # Joint projection
        joint_embeddings = self.joint_projection(joint_features)  # (batch_size * num_joints, hidden_nf)
        joint_embeddings = joint_embeddings.reshape(current_batch_size, self.num_joints, self.hidden_nf)
        
        # Object projection
        object_embeddings = self.object_projection(object_features)  # (batch_size, hidden_nf)
        object_embeddings = object_embeddings.reshape(current_batch_size, self.num_objects, self.hidden_nf)  # (batch_size, num_objects, hidden_nf)
        
        # Concatenate joint and object embeddings to create node sequence
        # Shape: (batch_size, num_joints + num_objects, hidden_nf)
        node_sequence = torch.cat([joint_embeddings, object_embeddings], dim=1)
        
        # Apply transformer
        transformer_output = self.transformer(node_sequence)  # (batch_size, num_joints + num_objects, hidden_nf)
        
        # Flatten and pass through output MLP
        flattened_output = transformer_output.reshape(current_batch_size, -1)  # (batch_size, (num_joints + num_objects) * hidden_nf)
        actions = self.output_mlp(flattened_output)  # (batch_size, num_joints)
        
        return actions
