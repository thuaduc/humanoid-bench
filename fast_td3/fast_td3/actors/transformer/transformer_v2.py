import torch
import torch.nn as nn

from fast_td3.robots.graph_builder import GraphBuilder


class DecoderLayer(nn.Module):
    """
    Single decoder-style transformer layer with pre-norm:
      Pre-norm → Self-attention (joints attend to each other)
      Pre-norm → Cross-attention (joints query object/goal token)
      Pre-norm → FFN (4× width)
    """

    def __init__(self, hidden_nf: int, num_heads: int, dropout: float, act_fn: nn.Module):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_nf)
        self.self_attn = nn.MultiheadAttention(
            hidden_nf, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_nf)
        self.cross_attn = nn.MultiheadAttention(
            hidden_nf, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_nf)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf * 4),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_nf * 4, hidden_nf),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self-attention (pre-norm)
        h = self.self_attn_norm(x)
        h, _ = self.self_attn(h, h, h)
        x = x + self.drop(h)

        # Cross-attention: joints query object token (pre-norm)
        h = self.cross_attn_norm(x)
        h, _ = self.cross_attn(h, context, context)
        x = x + self.drop(h)

        # FFN (pre-norm)
        x = x + self.ffn(self.ffn_norm(x))

        return x


class TransformerV2(nn.Module):
    """
    Decoder-style transformer actor: joints attend to each other AND query the
    object/goal token at every layer (not just at the end).

    Architecture per layer:
      Pre-norm → Self-attention (joints only)
      Pre-norm → Cross-attention (joints query goal token)
      Pre-norm → FFN (4× hidden)

    Compared to the previous version:
    - Goal conditioning at every layer (not just a single cross-attn after all self-attn)
    - Pre-norm for training stability
    - FFN width 4× (standard transformer practice)
    - act_fn used consistently throughout
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

        # Joint projection: in_joint_nf → hidden_nf
        self.joint_projection = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf),
            act_fn,
        )

        # Object projection: in_object_nf → hidden_nf
        self.object_projection = nn.Sequential(
            nn.Linear(in_object_nf, hidden_nf),
            act_fn,
        )

        # Learnable positional embeddings for joints
        self.positional_embeddings = nn.Parameter(
            torch.randn(self.num_joints, hidden_nf) * 0.02
        )

        # n decoder layers: each does self-attn + cross-attn + FFN
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_nf, num_heads, dropout, act_fn)
            for _ in range(n_layers)
        ])

        # Final layer-norm before action head
        self.final_norm = nn.LayerNorm(hidden_nf)

        # Per-joint action head
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
        batch_size = obs.shape[0]
        h_objects = obs[:, :self.object_feature_dim].unsqueeze(1)
        h_joints = obs[:, self.object_feature_dim:].view(
            batch_size, self.num_joints, self.in_joint_nf
        )

        # Project to hidden dimension
        joint_embeddings = self.joint_projection(h_joints)
        object_embeddings = self.object_projection(h_objects)  # (B, 1, hidden_nf)

        # Add positional embeddings to joints
        joint_embeddings = joint_embeddings + self.positional_embeddings.unsqueeze(0)

        # Pass through decoder layers (goal context at every layer)
        x = joint_embeddings
        for layer in self.layers:
            x = layer(x, object_embeddings)

        x = self.final_norm(x)

        # Per-joint action
        actions = self.action_head(x).squeeze(-1)
        return actions
