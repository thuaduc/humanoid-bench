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
        
        # Positional embeddings for joints — initialised with kinematic depth
        # sinusoidal encoding so joints at the same chain depth share similar codes.
        if robot == "h1" and self.num_joints == 19:
            import math
            from fast_td3.robots.h1 import H1
            h1_robot = H1()
            depth_map = h1_robot.kinematic_depth
            depths = [depth_map[H1.JOINT(j)] for j in range(self.num_joints)]
            pos_init = TransformerV2._make_sinusoidal_v2(depths, hidden_nf)
        else:
            pos_init = torch.randn(self.num_joints, hidden_nf) * 0.02
        self.positional_embeddings = nn.Parameter(pos_init)

        # Type embedding to mark joint tokens
        self.joint_type_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_nf) * 0.02
        )
        
        # Transformer encoder for joints only
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_nf,
            nhead=num_heads,
            dim_feedforward=hidden_nf * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layers,
        )
        
        # --- Soft kinematic attention bias for joint self-attention ---
        self.use_kinematic_bias = robot == "h1" and self.num_joints == 19
        if self.use_kinematic_bias:
            from fast_td3.robots.h1 import H1
            h1_robot = H1()
            raw_dist = h1_robot.get_distance_matrix()  # 19×19
            self.num_dist_buckets = 4
            dist_mat = torch.zeros(self.num_joints, self.num_joints, dtype=torch.long)
            for i in range(self.num_joints):
                for j in range(self.num_joints):
                    dist_mat[i, j] = min(raw_dist[i][j], self.num_dist_buckets - 1)
            self.register_buffer("dist_mat", dist_mat)
            self.kinematic_bias = nn.Parameter(torch.zeros(self.num_dist_buckets))

        # Cross-attention: joints query, object provides context
        self.joint_to_object_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_nf,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_nf)
        
        # Per-joint action head: hidden_nf -> hidden_nf -> 1
        self.action_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Tanh(),
        )
        
        self.to(device)

    @staticmethod
    def _make_sinusoidal_v2(depths: list, d_model: int) -> torch.Tensor:
        import math
        positions = torch.tensor(depths, dtype=torch.float32)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10.0) / max(d_model - 1, 1))
        )
        pe = torch.zeros(len(depths), d_model)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term.unsqueeze(0))
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term.unsqueeze(0))
        else:
            pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term[:-1].unsqueeze(0))
        return pe

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
        if self.use_kinematic_bias:
            attn_bias = self.kinematic_bias[self.dist_mat].to(obs.dtype)
        else:
            attn_bias = None
        joint_output = self.transformer(joint_embeddings, mask=attn_bias)
        
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
