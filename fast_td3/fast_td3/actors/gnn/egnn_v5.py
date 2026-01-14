from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egcl import E_GCL, env_with_object
from humanoid_bench.envs.custom_env import unflatten_obs


class JointObjectCrossAttention(nn.Module):
    """
    Cross-attention module for joint-object interaction.
    
    Joints (queries) attend to objects (keys/values) to capture
    spatial relationships between robot joints and task objects.
    """
    
    def __init__(self, joint_dim, object_dim, num_heads=1, dropout=0.0):
        """
        Args:
            joint_dim: Dimensionality of joint features (hidden_nf)
            object_dim: Dimensionality of object features (13: x, quat, vel)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention
        """
        super().__init__()
        
        # Embed object features to match joint dimension
        self.object_embedding = nn.Linear(object_dim, joint_dim)
        
        # Multi-head cross-attention: joints query objects
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            kdim=joint_dim,  # After embedding
            vdim=joint_dim,  # After embedding
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(joint_dim)
        
    def forward(self, joint_features, object_features):
        """
        Args:
            joint_features: (batch_size, num_joints, joint_dim)
            object_features: (batch_size, num_objects, object_dim)
            
        Returns:
            Enhanced joint features: (batch_size, num_joints, joint_dim)
        """
        # Embed objects to joint dimension
        object_embedded = self.object_embedding(object_features)  # (B, num_obj, joint_dim)
        
        # Cross-attention: joints attend to objects
        attn_out, _ = self.cross_attn(
            query=joint_features,
            key=object_embedded,
            value=object_embedded,
        )
        
        # Residual connection + normalization
        enhanced = self.norm(joint_features + attn_out)
        
        return enhanced


class EGNN_V5(nn.Module):
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
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        coords_agg="mean",
        coord_norm=False,
        extra_state_dim=0,
        num_attn_heads=1,
        attn_dropout=0.01,
    ):
        """
        Args:
            in_joint_nf: Number of features for joint nodes (velocity + position)
            in_object_nf: Number of features for object nodes (13: x, quat, vel)
            out_node_nf: Output node features (not used, kept for compatibility)
            hidden_nf: Number of hidden features (kept constant throughout)
            device: Device (e.g. 'cpu', 'cuda:0')
            batch_size: Batch size for edge caching
            act_fn: Non-linearity activation function
            n_layers: Number of E_GCL layers for joint message passing
            robot: Robot type ('h1', 'h1hand', etc.)
            env_name: Environment name for object determination
            residual: Use residual connections in E_GCL
            attention: Use attention in E_GCL
            normalize: Normalize coordinates in E_GCL
            tanh: Apply tanh to coordinate updates in E_GCL
            coords_agg: Coordinate aggregation method
            coord_norm: Apply coordinate normalization
            extra_state_dim: Extra object features (e.g., 6 for reach target, 12 for push)
            num_attn_heads: Number of attention heads (default: 1 for efficiency)
            attn_dropout: Dropout for cross-attention (default: 0.0)
        
        Note:
            For best performance, wrap model with torch.compile:
            model = torch.compile(model, mode='reduce-overhead')
        """
        super(EGNN_V5, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.batch_size = batch_size
        self.env_name = env_name
        self.robot = robot
        self.in_object_nf = in_object_nf
        self.extra_state_dim = extra_state_dim
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_objects = 2 if self.env_name in env_with_object else 1
        self._joint_edges_cache = {}
        
        # Total object feature dimension (base 13 + extra like reach target)
        self.total_object_nf = self.in_object_nf + (self.extra_state_dim // self.num_objects if self.extra_state_dim > 0 else 0)
        
        # Joint message passing layers (E_GCL)
        self.joint_layers = nn.ModuleList(
            [
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=0,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                    coords_agg=coords_agg,
                    coord_norm=coord_norm,
                )
                for _ in range(n_layers)
            ]
        )
        
        # Input embedding for joints
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, self.hidden_nf), 
            act_fn
        )
        
        # Cross-attention for object-joint interaction
        self.cross_attention = JointObjectCrossAttention(
            joint_dim=self.hidden_nf,
            object_dim=self.total_object_nf,
            num_heads=num_attn_heads,
            dropout=attn_dropout,
        )
        
        # Simplified 2-layer action head (replaces V3's 5-layer MLP)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_nf, 1),  # One action per joint
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention integration.
        Optimized to reduce reshaping operations.
        
        Args:
            obs: Flat observation tensor (batch_size, obs_dim)
            
        Returns:
            Actions: (batch_size, num_joints)
        """
        # Unflatten observation into components
        obs = unflatten_obs(obs, self.env_name)
        current_batch_size = obs["joint_velocities"].shape[0]
        joint_edges = self.get_cached_joint_edges(current_batch_size)

        # Prepare joint inputs in (batch, num_joints, feat) format
        h_joints = torch.stack(
            [obs["joint_velocities"], obs["joint_positions"]],
            dim=-1
        )  # (batch, 19, 2)
        h_joints = self.joint_embedding_in(h_joints)  # (batch, 19, hidden_nf)
        
        # Prepare object features
        h_objects = obs["object_features"]  # (batch, num_obj, 13)
        
        # Flatten for E_GCL message passing (only reshape once)
        h_joints_flat = h_joints.reshape(-1, self.hidden_nf)  # (batch*19, hidden_nf)
        x_joints_flat = obs["joint_x"].reshape(-1, 3)  # (batch*19, 3)

        # Message passing within joints (E_GCL layers)
        for layer in self.joint_layers:
            h_joints_flat, x_joints_flat, _ = layer(h=h_joints_flat, edge_index=joint_edges, coord=x_joints_flat)

        # Reshape back once for cross-attention
        h_joints = h_joints_flat.reshape(current_batch_size, self.num_joints, self.hidden_nf)
        
        # Cross-attention: joints attend to objects
        h_joints_enhanced = self.cross_attention(h_joints, h_objects)
        
        # Action head: (batch, 19, hidden_nf) -> (batch, 19)
        actions = self.action_head(h_joints_enhanced).squeeze(-1)
        
        return actions

    def generate_joint_edges(self, batch_size: int):
        """Generate batched edge indices for the joint graph."""
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        src = torch.tensor(src, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst, dtype=torch.long, device=self.device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=self.device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])
    
    def get_cached_joint_edges(self, current_batch_size: int):
        """Get cached edge indices for the joint graph."""
        if current_batch_size in self._joint_edges_cache:
            return self._joint_edges_cache[current_batch_size]
        
        edges = self.generate_joint_edges(current_batch_size)
        self._joint_edges_cache[current_batch_size] = edges
        return edges
