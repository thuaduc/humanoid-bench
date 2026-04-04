import math

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Transformer actor: flat observations are split by env-specific unflatten_obs into
    - h_objects: (B, 1, D_obj) — task-global context (pelvis, props, goals; D_obj varies by env)
    - h_joints: (B, 19, 5) — per joint [qpos, qvel, anchor_x, anchor_y, anchor_z] (anchors
      relative to pelvis, scaled like other custom envs)

    Forward: project both streams, concatenate as [joint_tokens, object_token], run
    TransformerEncoder, then read the first num_joints outputs for actions.
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
        :param in_joint_nf: Features per joint node (pos, vel, anchor xyz) = 5
        :param in_object_nf: Flat prefix length for the single object/global token (env-specific)
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

        # Joint projection: 5d -> hidden_nf
        self.joint_projection = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf),
            act_fn,
        )

        self.object_projection = nn.Sequential(
            nn.Linear(in_object_nf, hidden_nf),
            act_fn,
        )

        self.num_joints = 69 if "h1hand" in env_name else 19
        num_tokens = self.num_joints + 1  # joints + 1 object token

        # --- Topology-aware positional embeddings ---
        # For H1 (19 joints): initialise with sinusoidal encoding of kinematic
        # depth from torso so joints at the same chain depth share similar codes.
        # For other robots: fall back to small-random init.
        if robot == "h1" and self.num_joints == 19:
            from fast_td3.robots.h1 import H1
            self._h1 = H1()
            depth_map = self._h1.kinematic_depth
            depths = [depth_map[H1.JOINT(j)] for j in range(self.num_joints)]
            depths.append(max(depths) + 1)  # object token gets one level deeper
            pos_init = self._make_sinusoidal(depths, hidden_nf)
        else:
            pos_init = torch.randn(num_tokens, hidden_nf) * 0.02
        self.positional_embeddings = nn.Parameter(pos_init)

        # --- Soft kinematic attention bias (ALiBi-style) ---
        # For H1: add a learnable scalar per BFS-distance bucket to attention
        # logits so kinematic neighbours are naturally preferred.
        # Initialised at 0 → standard full attention at the start of training.
        self.use_kinematic_bias = robot == "h1" and self.num_joints == 19
        if self.use_kinematic_bias:
            h1 = getattr(self, "_h1", None) or H1()
            raw_dist = h1.get_distance_matrix()  # List[List[int]], 19×19
            self.num_dist_buckets = 4  # buckets: 0 (self), 1, 2, 3+ (far)
            dist_mat = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
            for i in range(self.num_joints):
                for j in range(self.num_joints):
                    dist_mat[i, j] = min(raw_dist[i][j], self.num_dist_buckets - 1)
            # Object-token row/column stays at bucket 0 → no topology restriction
            self.register_buffer("dist_mat", dist_mat)
            self.kinematic_bias = nn.Parameter(torch.zeros(self.num_dist_buckets))

        # Transformer encoder
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

        # Per-joint action head: hidden_nf -> 1  (slim head; experiments show
        # the large 4× head hurts more than it helps)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Tanh(),
        )

        self.to(device)

    @staticmethod
    def _make_sinusoidal(depths: list, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding keyed on kinematic depth."""
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
        batch_size = obs.shape[0]
        h_objects = obs[:, :self.in_object_nf].unsqueeze(1)
        h_joints = obs[:, self.in_object_nf:].view(batch_size, self.num_joints, self.in_joint_nf)

        joint_embeddings = self.joint_projection(h_joints)
        object_embeddings = self.object_projection(h_objects)

        # Sequence: joints 0..N-1, then global object token
        node_sequence = torch.cat([joint_embeddings, object_embeddings], dim=1)
        node_sequence = node_sequence + self.positional_embeddings.unsqueeze(0)

        # Build additive attention bias from kinematic graph distances
        if self.use_kinematic_bias:
            attn_bias = self.kinematic_bias[self.dist_mat].to(obs.dtype)
        else:
            attn_bias = None
        transformer_output = self.transformer(node_sequence, mask=attn_bias)

        joint_output = transformer_output[:, : self.num_joints, :]
        actions = self.action_head(joint_output)
        actions = actions.squeeze(-1)
        
        return actions