from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egcl import E_GCL
from humanoid_bench.envs.custom_env import unflatten_obs, get_env_class


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
        object_embedded = self.object_embedding(object_features)  # (B, joint_dim)
        
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
        num_attn_heads=1,
        attn_dropout=0.01,
    ):
        """
        Args:
            in_joint_nf: Number of features for joint nodes (velocity + position)
            in_object_nf: Number of features for object nodes (13: x, quat, vel)
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
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        env_class = get_env_class(env_name)
        self.num_objects = env_class.num_objects
        
        # Pre-generate edge caches for common batch sizes to avoid dynamic dict access
        # This is crucial for torch.compile with CUDA graphs
        # Register as buffers for proper device handling and state persistence
        common_batch_sizes = [128, batch_size]
        for bs in common_batch_sizes:
            self.register_buffer(
                f'_joint_edges_{bs}',
                self.generate_joint_edges(bs),
                persistent=True
            )
        
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
            object_dim=self.in_object_nf,
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
        h_joints, x_joints_flat, h_objects = unflatten_obs(obs, self.env_name)
        current_batch_size = h_objects.shape[0]
        joint_edges = self.get_cached_joint_edges(current_batch_size)

        # Reshape h_joints from (batch*19, 2) to (batch, 19, 2) for embedding
        h_joints = h_joints.reshape(current_batch_size, self.num_joints, 2)
        h_joints = self.joint_embedding_in(h_joints)  # (batch, 19, hidden_nf)
        
        # Prepare object features: (batch, 1, in_object_nf)
        h_objects = h_objects.unsqueeze(1)
        
        # Flatten for E_GCL message passing
        h_joints_flat = h_joints.reshape(-1, self.hidden_nf)  # (batch*19, hidden_nf)
        # x_joints_flat already in correct shape (batch*19, 3)

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
        # Use cached edges if available, otherwise generate on-the-fly
        # Note: For best compile performance, ensure batch_size is pre-cached during init
        buffer_name = f'_joint_edges_{current_batch_size}'
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)
        
        # Fallback for uncached batch sizes (warning: may break CUDA graphs)
        return self.generate_joint_edges(current_batch_size)
