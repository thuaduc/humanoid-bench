import torch
import torch.nn as nn

from fast_td3.robots.graph_builder import GraphBuilder
from humanoid_bench.envs.custom_env import unflatten_obs


class Transformer(nn.Module):
    """
    Transformer-based actor with skip connections and explicit object conditioning.
    
    Architecture:
    1. Project joint (5d) and object (13d) features to 64d
    2. Add learnable positional embeddings
    3. Apply 2-layer transformer (64d, 4 heads)
    4. Add object embedding to each joint (explicit conditioning)
    5. Concatenate transformer output (64d) with raw joint features (5d) = 69d
    6. Per-joint action head: 69d -> 64d -> 32d -> 1
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
        :param in_joint_nf: Number of features for joint nodes (pos, vel, x, y, z) = 5
        :param in_object_nf: Number of features for object nodes (pelvis state) = 13
        :param out_node_nf: Output dimension per node (unused in this architecture)
        :param hidden_nf: Hidden dimension for embeddings and transformer (64d)
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param batch_size: Batch size for vectorized environments
        :param act_fn: Activation function
        :param n_layers: Number of transformer layers (2)
        :param robot: Robot name for graph builder
        :param env_name: Environment name
        :param num_heads: Number of attention heads (4)
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
        
        # Joint projection: 5d -> hidden_nf (64d)
        self.joint_projection = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf),
            act_fn,
        )
        
        # Object projection: 13d -> hidden_nf (64d)
        self.object_projection = nn.Sequential(
            nn.Linear(in_object_nf, hidden_nf),
            act_fn,
        )
        
        # Learnable positional embeddings for joints and objects
        # We'll have (num_joints + 1) positions since we have 1 object token
        max_positions = self.num_joints + 1
        self.positional_embeddings = nn.Parameter(
            torch.randn(max_positions, hidden_nf) * 0.02
        )
        
        # Transformer encoder: 2 layers, 64d, 4 heads
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_nf,
            nhead=num_heads,
            dim_feedforward=hidden_nf * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layers,
        )
        
        # Per-joint action head: 69d -> 64d -> 32d -> 1
        # Input is contextualized features (64d) + raw features (5d) = 69d
        self.action_head = nn.Sequential(
            nn.Linear(hidden_nf + in_joint_nf, 64),
            act_fn,
            nn.Linear(64, 32),
            act_fn,
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observations through the transformer with skip connections.
        
        Architecture flow:
        1. Project joint (5d) and object (13d) features to 64d
        2. Add positional embeddings
        3. Apply transformer (2 layers, 64d, 4 heads)
        4. Add object embedding to each joint (explicit conditioning)
        5. Concatenate contextualized (64d) + raw features (5d)
        6. Per-joint action head (69d -> 1d)
        
        :param obs: Flattened observation tensor
        :return: Action tensor of shape (batch_size, num_joints)
        """
        h_joints, h_objects = unflatten_obs(obs, self.env_name)
        current_batch_size = h_joints.shape[0]

        # Step 1: Project features to hidden dimension
        joint_embeddings = self.joint_projection(h_joints)
        object_embeddings = self.object_projection(h_objects)
        
        # Step 2: Add positional embeddings
        node_sequence = torch.cat([joint_embeddings, object_embeddings], dim=1)
        node_sequence = node_sequence + self.positional_embeddings.unsqueeze(0)
        
        # Step 3: Apply transformer
        transformer_output = self.transformer(node_sequence)
        
        # Step 4: Split back into joints and objects
        h_joint_ctx = transformer_output[:, :self.num_joints, :]
        h_object_ctx = transformer_output[:, self.num_joints:, :]
        
        # Step 5: Explicit object conditioning - add object embedding to each joint
        h_object_broadcasted = h_object_ctx.mean(dim=1, keepdim=True)
        h_joint_conditioned = h_joint_ctx + h_object_broadcasted
        
        # Step 6: Skip connection - concat contextualized (64d) + raw features (5d)
        h_joint_with_skip = torch.cat([h_joint_conditioned, h_joints], dim=-1)
        
        # Step 7: Per-joint action head (69d -> 1d)
        # Process each joint independently through the same action head
        actions = self.action_head(h_joint_with_skip)
        actions = actions.squeeze(-1)
        
        return actions
