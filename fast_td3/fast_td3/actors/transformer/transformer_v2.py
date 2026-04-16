import torch
import torch.nn as nn

class TransformerV2(nn.Module):
    """
    Transformer-based actor with hierarchical attention: joints-only encoder + cross-attention.
    
    Architecture:
    1. Project joint (5d) and object features to hidden_nf
    2. Add type embeddings (joint vs object tokens) + positional embeddings to joints only
    3. Apply n-layer self-attention transformer to joints only
    4. Cross-attention: joints query object node for global context
    5. Global action head: flattened joint features -> full action vector (shared weights, no per-joint split MLP)
    
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
        n_act,
        num_heads=4,
        dropout=0.1,
    ):
        """
        :param in_joint_nf: Number of features for joint nodes (pos, vel, x, y, z) = 5
        :param in_object_nf: Number of features for object/global token (env-specific)
        :param out_node_nf: Output dimension per node (unused in this architecture)
        :param n_act: Action dimension (action head output size)
        :param hidden_nf: Hidden dimension for embeddings and transformer
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param batch_size: Batch size for vectorized environments
        :param act_fn: Activation function
        :param n_layers: Number of transformer layers
        :param robot: Robot name for graph builder
        :param env_name: Environment name (selects joint count: 69 for h1hand, else 19)
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
        self.n_act = n_act
        self.num_heads = num_heads
        self.num_joints = 69 if "h1hand" in env_name else 19
        
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
        # self.query_proj = nn.Linear(hidden_nf, hidden_nf)
        # self.key_proj = nn.Linear(hidden_nf, hidden_nf)
        # self.value_proj = nn.Linear(hidden_nf, hidden_nf)
        # self.out_proj = nn.Linear(hidden_nf, hidden_nf)
        # self.cross_attn_norm = nn.LayerNorm(hidden_nf)
        # self.cross_attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Project contextualized features back to per-joint feature size before action head
        self.embedding_out = nn.Sequential(
            nn.Linear(hidden_nf, in_joint_nf),
            act_fn,
        )
        
        # Single MLP on all joint features at once (same parameters for the full body)
        action_in_dim = self.num_joints * in_joint_nf + in_object_nf
        
        self.action_head = nn.Sequential(
            nn.Linear(action_in_dim, 512),
            act_fn,
            nn.Linear(512, 128),
            act_fn,
            nn.Linear(128, n_act),
            nn.Tanh(),
        )
        
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observations through joints-only transformer.
        
        Architecture:
        1. Project joint features to hidden dimension
        2. Add positional embeddings to joints
        3. Apply self-attention transformer to joints only
        4. Project back to joint feature size
        5. Concatenate with object features and pass through action head
        """
        batch_size = obs.shape[0]
        
        # Split observation: [object_features, joint_features_flat]
        h_objects = obs[:, :self.object_feature_dim]
        h_joints = obs[:, self.object_feature_dim:].view(batch_size, self.num_joints, self.in_joint_nf)

        # Project joint features and add positional embeddings
        joint_embeddings = self.joint_projection(h_joints)
        joint_embeddings = joint_embeddings + self.positional_embeddings
        
        # Apply transformer encoder
        joint_output = self.transformer(joint_embeddings)
        
        # Project back to joint feature size and flatten
        joint_output = self.embedding_out(joint_output)
        joint_flat = joint_output.view(batch_size, -1)
        
        # Concatenate joint and object features
        combined_features = torch.cat([joint_flat, h_objects], dim=-1)
        
        # Generate actions
        actions = self.action_head(combined_features)
        
        return actions
