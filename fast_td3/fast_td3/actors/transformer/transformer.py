import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Transformer actor: flat observations are split by env-specific unflatten_obs into
    - h_objects: (B, 1, D_obj) — task-global context (pelvis, props, goals; D_obj varies by env)
    - h_joints: (B, 19, 5) — per joint [qpos, qvel, anchor_x, anchor_y, anchor_z] (anchors
      relative to pelvis, scaled like other custom envs)

    Forward: project both streams, concatenate as [joint_tokens, object_token], run
    TransformerEncoder, project joint outputs back to in_joint_nf, then one global MLP on
    flattened joint features produces the full action vector.
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
        :param in_joint_nf: Features per joint node (pos, vel, anchor xyz) = 5
        :param in_object_nf: Flat prefix length for the single object/global token (env-specific)
        :param out_node_nf: Output dimension per node (unused in this architecture)
        :param n_act: Action dimension (action head output size)
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
        self.n_act = n_act
        self.num_heads = num_heads
        
        # Joint projection: 5d -> hidden_nf (64d)
        self.joint_projection = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf),
            act_fn,
        )
        
        self.object_projection = nn.Sequential(
            nn.Linear(in_object_nf, hidden_nf),
            act_fn,
        )
        
        # Learnable positional embeddings for joints and objects
        # We'll have (num_joints + 1) positions since we have 1 object token
        self.num_joints = 69 if "h1hand" in env_name else 19
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
        
        self.embedding_out = nn.Sequential(
            nn.Linear(hidden_nf, in_joint_nf),
            act_fn,
        )
        
        action_in_dim = self.num_joints * in_joint_nf
        self.action_head = nn.Sequential(
            nn.Linear(action_in_dim, 256),
            act_fn,
            nn.Linear(256, 128),
            act_fn,
            nn.Linear(128, n_act),
            nn.Tanh(),
        )
        
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        h_objects = obs[:, :self.in_object_nf].unsqueeze(1)
        h_joints = obs[:, self.in_object_nf:].view(batch_size, self.num_joints, self.in_joint_nf)

        joint_embeddings = self.joint_projection(h_joints)
        object_embeddings = self.object_projection(h_objects)

        # Sequence layout matches positional_embeddings: joints 0..N-1, global token last
        node_sequence = torch.cat([joint_embeddings, object_embeddings], dim=1)
        node_sequence = node_sequence + self.positional_embeddings.unsqueeze(0)

        transformer_output = self.transformer(node_sequence)
        transformer_output = transformer_output + self.positional_embeddings.unsqueeze(0)

        joint_output = transformer_output[:, : self.num_joints, :]
        joint_output = self.embedding_out(joint_output)
        actions = self.action_head(joint_output.reshape(batch_size, -1))
        
        return actions