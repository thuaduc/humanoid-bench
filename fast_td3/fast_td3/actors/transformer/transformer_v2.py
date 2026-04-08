import torch
import torch.nn as nn

from fast_td3.robots.graph_builder import GraphBuilder


class TransformerV2(nn.Module):
    """
    Transformer-based actor with hierarchical attention: joints-only encoder + cross-attention.
    
    Architecture:
    1. Project joint (5d) and object features to hidden_nf
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
        :param in_object_nf: Number of features for object/global token (env-specific)
        :param out_node_nf: Output dimension per node (unused in this architecture)
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
        self.object_feature_dim = in_object_nf
        self.num_heads = num_heads
        
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        
        # Joint projection: 5d -> hidden_nf
        self.joint_projection = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf),
            act_fn,
        )
        
        # Object projection: object_feature_dim -> hidden_nf
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
        
        # Transformer encoder for joints only
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
        # Use manual implementation to avoid CUDA errors with torch.compile + single-token sequences
        self.query_proj = nn.Linear(hidden_nf, hidden_nf)
        self.key_proj = nn.Linear(hidden_nf, hidden_nf)
        self.value_proj = nn.Linear(hidden_nf, hidden_nf)
        self.out_proj = nn.Linear(hidden_nf, hidden_nf)
        self.cross_attn_norm = nn.LayerNorm(hidden_nf)
        self.cross_attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Per-joint action head: hidden_nf -> hidden_nf -> 1
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
        batch_size = obs.shape[0]
        h_objects = obs[:, :self.object_feature_dim].unsqueeze(1)
        h_joints = obs[:, self.object_feature_dim:].view(batch_size, self.num_joints, self.in_joint_nf)

        # Step 1: Project features to hidden dimension
        joint_embeddings = self.joint_projection(h_joints)
        object_embeddings = self.object_projection(h_objects)
        
        # Step 2: Add type and positional embeddings to joints only
        joint_embeddings = joint_embeddings + self.joint_type_embedding
        joint_embeddings = joint_embeddings + self.positional_embeddings.unsqueeze(0)
        
        # Step 3: Apply self-attention transformer to joints only
        joint_output = self.transformer(joint_embeddings)
        
        # Step 4: Cross-attention - joints query object for global context
        # Manual implementation to avoid CUDA errors with torch.compile on single-token sequences
        Q = self.query_proj(joint_output)  # (B, num_joints, hidden_nf)
        K = self.key_proj(object_embeddings)  # (B, 1, hidden_nf)
        V = self.value_proj(object_embeddings)  # (B, 1, hidden_nf)
        
        # Compute attention: Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_nf ** 0.5)  # (B, num_joints, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, num_joints, 1)
        attn_weights = self.cross_attn_dropout(attn_weights)
        
        # Apply attention to values
        joint_contextualized = torch.matmul(attn_weights, V)  # (B, num_joints, hidden_nf)
        joint_contextualized = self.out_proj(joint_contextualized)
        
        joint_output = self.cross_attn_norm(joint_output + joint_contextualized)
        
        # Step 5: Generate actions from contextualized joint features
        actions = self.action_head(joint_output)
        actions = actions.squeeze(-1)
        
        return actions
