from torch import nn
import torch

from humanoid_bench.envs.custom_env import unflatten_obs


class GATLayer(nn.Module):
    """Single GAT layer with multi-head self-attention."""
    
    def __init__(self, hidden_dim, num_heads, dropout, residual=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual = residual
    
    def forward(self, h_joints):
        """
        Args:
            h_joints: (batch, num_joints, hidden_dim)
            
        Returns:
            h_joints: (batch, num_joints, hidden_dim)
        """
        # Fully connected self-attention
        attn_out, _ = self.attn(h_joints, h_joints, h_joints)
        
        if self.residual:
            return self.norm(h_joints + attn_out)
        return self.norm(attn_out)


class GAT(nn.Module):
    """GAT network for joint processing with concat + MLP architecture."""
    
    def __init__(
        self,
        n_act,
        in_joint_nf,
        hidden_nf,
        object_feature_dim,
        n_layers,
        num_heads,
        dropout,
        act_fn,
        device,
        batch_size,
        env_name,
        robot,
        residual=True,
        joint_out_dim=2,
    ):
        """
        Args:
            in_joint_nf: Number of input features per joint (5: pos, vel, x, y, z)
            hidden_nf: Hidden dimension for GAT layers
            object_feature_dim: Dimension of object features
            n_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability for attention
            act_fn: Activation function
            device: Device to place the model on
            batch_size: Batch size (unused, kept for API consistency)
            env_name: Environment name
            robot: Robot type (unused, kept for API consistency)
            residual: Use residual connections in GAT layers
            joint_out_dim: Embedding dimension before concatenation
        """
        super().__init__()
        self.env_name = env_name
        self.device = device
        self.num_joints = 19 if 'h1hand' not in env_name else 69
        self.n_act = n_act
        
        # Joint embedding: (batch, num_joints, 5) -> (batch, num_joints, hidden_dim)
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, hidden_nf), act_fn
        )
        
        # Stack of GAT layers for message passing
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_nf, num_heads, dropout, residual)
            for _ in range(n_layers)
        ])
        
        # Embed down: (batch, num_joints, hidden_dim) -> (batch, num_joints, joint_out_dim)
        self.joint_embedding_out = nn.Sequential(
            nn.Linear(hidden_nf, joint_out_dim), act_fn
        )
        
        # Combined dimension after concatenation
        joint_object_dim = joint_out_dim * self.num_joints + object_feature_dim
        
        # Large MLP action head (like EGNN v3 pattern)
        self.action_head = nn.Sequential(
            nn.Linear(joint_object_dim, hidden_nf * 4),
            act_fn,
            nn.Linear(hidden_nf * 4, hidden_nf * 2),
            act_fn,
            nn.Linear(hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.n_act),
            nn.Tanh(),
        )
        
        self.to(device)
    
    def forward(self, obs):
        """
        Forward pass through GAT network.
        
        Args:
            obs: Flat observation tensor (batch_size, obs_dim)
            
        Returns:
            Actions: (batch_size, num_joints)
        """
        # Unflatten: returns 2 values (h_joints, h_objects)
        h_joints, h_objects = unflatten_obs(obs, self.env_name)
        # h_joints: (batch, num_joints, 5), h_objects: (batch, 1, obj_dim)
        
        # Embed joints to hidden dimension
        h_joints = self.joint_embedding_in(h_joints)  # (batch, num_joints, hidden_dim)
        
        # Self-attention layers (fully connected message passing)
        for layer in self.gat_layers:
            h_joints = layer(h_joints)
        
        # Embed down to lower dimension
        h_joints = self.joint_embedding_out(h_joints)  # (batch, num_joints, joint_out_dim)
        
        # Flatten joints
        h_joints_flat = h_joints.reshape(h_joints.shape[0], -1)  # (batch, num_joints*joint_out_dim)
        
        # Flatten objects (remove the singleton dimension)
        h_objects_flat = h_objects.squeeze(1)  # (batch, obj_dim)
        
        # Concatenate
        h_combined = torch.cat([h_joints_flat, h_objects_flat], dim=-1)
        
        # Generate actions through large MLP
        actions = self.action_head(h_combined)  # (batch, num_joints)
        return actions
