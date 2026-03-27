import torch
import torch.nn as nn

from fast_td3.robots.graph_builder import GraphBuilder
from humanoid_bench.envs.custom_env import unflatten_obs


class TransformerV2(nn.Module):
    """
    Transformer-based actor with hierarchical attention: joints-only encoder + cross-attention.
    
    Architecture:
    1. Project joint (5d) and object (13d) features to hidden_nf
    2. Add type embeddings (joint vs object tokens) + positional embeddings to joints only
    3. Apply n-layer self-attention transformer to joints only
    4. Cross-attention: joints query object node for global context
    5. Per-joint action head
    
    Key improvement: Joints first attend to each other, then explicitly query the object
    node for global context through cross-attention. This creates a clearer hierarchy where
    object information is integrated as context rather than being mixed equally in self-attention.
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
        
        # Learnable positional embeddings for joints only
        self.positional_embeddings = nn.Parameter(
            torch.randn(self.num_joints, hidden_nf) * 0.02
        )
        
        # Type embedding to mark joint tokens
        self.joint_type_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_nf) * 0.02
        )
        
        # Transformer encoder for joints only: 2 layers, 64d, 4 heads
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
        
        # Cross-attention: joints query, object provides context
        self.joint_to_object_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_nf,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_nf)
        
        # Per-joint action head: 64d -> 256d -> 64d -> 1
        self.action_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf * 4),
            act_fn,
            nn.Linear(hidden_nf * 4, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Tanh(),
        )
        
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observations through joints-only transformer + cross-attention.
        
        Architecture:
        1. Project features
        2. Add type and positional embeddings to joints only
        3. Apply self-attention transformer to joints only (no object in self-attention)
        4. Cross-attention: joints query object for global context
        5. Generate actions from contextualized joint features
        """
        h_joints, h_objects = unflatten_obs(obs, self.env_name)

        # Step 1: Project features to hidden dimension
        joint_embeddings = self.joint_projection(h_joints)
        object_embeddings = self.object_projection(h_objects)
        
        # Step 2: Add type and positional embeddings to joints only
        joint_embeddings = joint_embeddings + self.joint_type_embedding
        joint_embeddings = joint_embeddings + self.positional_embeddings.unsqueeze(0)
        
        # Step 3: Apply self-attention transformer to joints only
        joint_output = self.transformer(joint_embeddings)
        
        # Step 4: Cross-attention - joints query object for global context
        joint_contextualized, _ = self.joint_to_object_cross_attn(
            query=joint_output,
            key=object_embeddings,
            value=object_embeddings,
        )
        joint_output = self.cross_attn_norm(joint_output + joint_contextualized)
        
        # Step 5: Generate actions from contextualized joint features
        actions = self.action_head(joint_output)
        actions = actions.squeeze(-1)
        
        return actions
